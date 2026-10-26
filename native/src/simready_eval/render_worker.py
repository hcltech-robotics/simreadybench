"""Blender-only entrypoint for the neutral manifest; no OpenUSD/private imports."""
import json
from pathlib import Path
import sys
import struct

import bpy
from mathutils import Matrix, Vector


IMAGE_CACHE={}
_UV_FIELDS=frozenset(('diffuseColor','roughness','metallic','opacity','normal','normal_frame','emissiveColor'))


def _intern_uv_corners(value):
    # JSON repeats shared coordinate lists. Restore sharing only within one
    # triangle's UV dictionary, using exact double bits (including signed zero).
    if not value or not set(value).issubset(_UV_FIELDS):return value
    for coordinates in value.values():
        if not isinstance(coordinates,list)or len(coordinates)!=3:return value
        if any(not isinstance(corner,list)or len(corner)!=2 for corner in coordinates):return value
    pool={}
    for name,coordinates in value.items():
        key=struct.pack('<6d',coordinates[0][0],coordinates[0][1],coordinates[1][0],coordinates[1][1],coordinates[2][0],coordinates[2][1])
        value[name]=pool.setdefault(key,coordinates)
    return value


def _cpu_execution_settings(recipe):
    """Reject GPU labels and unsupported CPU sample/thread pairs before setup."""
    if not isinstance(recipe,dict) or recipe.get('schema')!='simready-neutral-cycles-v1' or recipe.get('engine')!='CYCLES' or recipe.get('device')!='CPU' or recipe.get('blender_version')!='4.3.2':
        raise ValueError('Expected a qualified CPU execution recipe')
    samples=recipe.get('samples');threads=recipe.get('threads',1)
    if type(samples)is not int or type(threads)is not int or (samples,threads)not in ((8,1),(256,4)):
        raise ValueError('Expected a qualified CPU sample/thread pair')
    return samples,threads


def texture(nodes, links, descriptor, uv_name):
    if descriptor.get('fallback_only'):
        outputs=[]
        for i,channel in enumerate(descriptor['channels']):
            constant=nodes.new('ShaderNodeValue')
            constant.outputs[0].default_value=descriptor['fallback'][channel]*descriptor['scale'][i]+descriptor['bias'][i]
            outputs.append(constant.outputs[0])
    else:
        uv=nodes.new('ShaderNodeUVMap');uv.uv_map=uv_name
        separate=nodes.new('ShaderNodeSeparateXYZ');links.new(uv.outputs['UV'],separate.inputs[0])
        vector=uv.outputs['UV']
        if not descriptor.get('udim'):
            combine=nodes.new('ShaderNodeCombineXYZ')
            for axis,mode in enumerate(descriptor['wrap']):
                operation=nodes.new('ShaderNodeMath');operation.operation='FRACT'if mode=='repeat'else'CLAMP'
                links.new(separate.outputs[axis],operation.inputs[0])
                if mode=='clamp':operation.inputs[1].default_value=0;operation.inputs[2].default_value=1
                links.new(operation.outputs[0],combine.inputs[axis])
            vector=combine.outputs[0]
        node=nodes.new('ShaderNodeTexImage')
        image_key=(descriptor['file'],descriptor['sRGB'])
        if image_key not in IMAGE_CACHE:
            file=descriptor['tiles'][min(descriptor['tiles'],key=int)] if descriptor.get('udim') else descriptor['file']
            image=bpy.data.images.load(file,check_existing=False)
            if descriptor.get('udim'):
                image.source='TILED';image.filepath=descriptor['file']
                existing={tile.number for tile in image.tiles}
                for tile in sorted(map(int,descriptor['tiles'])):
                    if tile not in existing:image.tiles.new(tile)
                image.reload()
            image.colorspace_settings.name='sRGB'if descriptor['sRGB']else'Non-Color'
            IMAGE_CACHE[image_key]=image
        node.image=IMAGE_CACHE[image_key];node.interpolation='Linear';node.extension='EXTEND'
        links.new(vector,node.inputs['Vector'])
        separate_color=nodes.new('ShaderNodeSeparateColor');links.new(node.outputs['Color'],separate_color.inputs[0])
        valid=None
        if descriptor.get('udim'):
            floors=[]
            for axis in (0,1):
                floor=nodes.new('ShaderNodeMath');floor.operation='FLOOR';links.new(separate.outputs[axis],floor.inputs[0]);floors.append(floor.outputs[0])
            for tile in sorted(map(int,descriptor['tiles'])):
                tests=[]
                for axis,coordinate in enumerate(((tile-1001)%10,(tile-1001)//10)):
                    test=nodes.new('ShaderNodeMath');test.operation='COMPARE';links.new(floors[axis],test.inputs[0]);test.inputs[1].default_value=coordinate;test.inputs[2].default_value=.01;tests.append(test.outputs[0])
                conjunction=nodes.new('ShaderNodeMath');conjunction.operation='MULTIPLY'
                links.new(tests[0],conjunction.inputs[0]);links.new(tests[1],conjunction.inputs[1])
                if valid is None:valid=conjunction.outputs[0]
                else:
                    add=nodes.new('ShaderNodeMath');add.operation='MAXIMUM';links.new(valid,add.inputs[0]);links.new(conjunction.outputs[0],add.inputs[1]);valid=add.outputs[0]
        outputs=[]
        for i,channel in enumerate(descriptor['channels']):
            source=node.outputs['Alpha']if channel==3 else separate_color.outputs[channel]
            if valid is not None:
                difference=nodes.new('ShaderNodeMath');difference.operation='SUBTRACT';links.new(source,difference.inputs[0]);difference.inputs[1].default_value=descriptor['fallback'][channel]
                select=nodes.new('ShaderNodeMath');select.operation='MULTIPLY_ADD';links.new(difference.outputs[0],select.inputs[0]);links.new(valid,select.inputs[1]);select.inputs[2].default_value=descriptor['fallback'][channel];source=select.outputs[0]
            multiply=nodes.new('ShaderNodeMath');multiply.operation='MULTIPLY_ADD'
            links.new(source,multiply.inputs[0]);multiply.inputs[1].default_value=descriptor['scale'][i];multiply.inputs[2].default_value=descriptor['bias'][i];outputs.append(multiply.outputs[0])
    if len(outputs)==1:return outputs[0]
    combine_color=nodes.new('ShaderNodeCombineColor')
    for i,output in enumerate(outputs[:3]):links.new(output,combine_color.inputs[i])
    return combine_color.outputs[0]


def make_material(record):
    material=bpy.data.materials.new('PreviewSurface'); material.use_nodes=True
    nodes=material.node_tree.nodes; nodes.clear(); links=material.node_tree.links
    output=nodes.new('ShaderNodeOutputMaterial'); shader=nodes.new('ShaderNodeBsdfPrincipled')
    shader.inputs['IOR'].default_value=record.get('ior',1.5)
    for name,socket in [('diffuseColor','Base Color'),('roughness','Roughness'),('metallic','Metallic')]:
        value=record[name]
        if isinstance(value,dict): links.new(texture(nodes,links,value,name),shader.inputs[socket])
        else: shader.inputs[socket].default_value=(*value,1) if name=='diffuseColor' else value
    if record.get('source_material_status')=='unbound':
        display=nodes.new('ShaderNodeVertexColor');display.layer_name='displayColor';links.new(display.outputs['Color'],shader.inputs['Base Color'])
    emission=record.get('emissiveColor',[0,0,0]);shader.inputs['Emission Strength'].default_value=1
    if isinstance(emission,dict):links.new(texture(nodes,links,emission,'emissiveColor'),shader.inputs['Emission Color'])
    else:shader.inputs['Emission Color'].default_value=(*emission,1)
    normal=record.get('normal')
    if isinstance(normal,dict):
        signed=texture(nodes,links,normal,'normal')
        encode=nodes.new('ShaderNodeVectorMath');encode.operation='MULTIPLY_ADD'
        links.new(signed,encode.inputs[0]);encode.inputs[1].default_value=(.5,.5,.5);encode.inputs[2].default_value=(.5,.5,.5)
        normalmap=nodes.new('ShaderNodeNormalMap');normalmap.space='TANGENT';normalmap.uv_map='normal_frame'
        normalmap.inputs['Strength'].default_value=1
        links.new(encode.outputs[0],normalmap.inputs['Color']);links.new(normalmap.outputs['Normal'],shader.inputs['Normal'])
    transparent=nodes.new('ShaderNodeBsdfTransparent'); mix=nodes.new('ShaderNodeMixShader')
    opacity=record['opacity']
    if isinstance(opacity,dict): links.new(texture(nodes,links,opacity,'opacity'),mix.inputs[0])
    else: mix.inputs[0].default_value=opacity
    if record.get('source_material_status')=='unbound':
        opacity_attribute=nodes.new('ShaderNodeVertexColor');opacity_attribute.layer_name='displayOpacity'
        opacity_channel=nodes.new('ShaderNodeSeparateColor');links.new(opacity_attribute.outputs['Color'],opacity_channel.inputs[0]);links.new(opacity_channel.outputs[0],mix.inputs[0])
    links.new(transparent.outputs[0],mix.inputs[1]); links.new(shader.outputs[0],mix.inputs[2])
    if record.get('double_sided',False):
        links.new(mix.outputs[0],output.inputs['Surface'])
    else:
        geometry=nodes.new('ShaderNodeNewGeometry'); backface=nodes.new('ShaderNodeMixShader')
        links.new(geometry.outputs['Backfacing'],backface.inputs[0])
        links.new(mix.outputs[0],backface.inputs[1]); links.new(transparent.outputs[0],backface.inputs[2])
        links.new(backface.outputs[0],output.inputs['Surface'])
    return material


def main():
    path=Path(sys.argv[sys.argv.index('--')+1]); data=json.loads(path.read_text(),object_hook=_intern_uv_corners)
    output=Path(data['output_dir']); recipe=data['recipe']; rig=data['cameras']
    samples,threads=_cpu_execution_settings(recipe)
    if bpy.app.version_string!=recipe['blender_version']:
        raise RuntimeError('Unqualified Blender version: '+bpy.app.version_string)
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.device='CPU'
    scene.cycles.samples=samples; scene.cycles.seed=recipe['seed']
    scene.cycles.use_adaptive_sampling=False; scene.cycles.use_denoising=False
    scene.render.threads_mode='FIXED'; scene.render.threads=threads
    scene.render.resolution_x,scene.render.resolution_y=rig['resolution']; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8'; scene.render.film_transparent=False
    scene.render.filter_size=recipe['pixel_filter_width']
    scene.render.dither_intensity=recipe['dither_intensity']
    scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'
    scene.view_settings.exposure=recipe['exposure']; scene.view_settings.gamma=recipe['gamma']
    scene.world=bpy.data.worlds.new('FrozenWorld'); scene.world.use_nodes=True
    background=scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value=(*recipe['background_linear_rgb'],1)
    background.inputs['Strength'].default_value=recipe['world_strength']
    for light in recipe['lights']:
        lamp=bpy.data.lights.new('FrozenArea','AREA'); lamp.energy=light['energy']; lamp.shape='DISK'; lamp.size=light['size']
        obj=bpy.data.objects.new('FrozenArea',lamp); scene.collection.objects.link(obj); obj.location=light['position']
        obj.rotation_euler=(-obj.location).to_track_quat('-Z','Y').to_euler()
    mesh=bpy.data.meshes.new('NeutralSurface')
    points=[point for triangle in data['triangles'] for point in triangle]
    mesh.from_pydata(points,[],[(i,i+1,i+2) for i in range(0,len(points),3)]); mesh.update()
    obj=bpy.data.objects.new('NeutralSurface',mesh); scene.collection.objects.link(obj)
    for record in data['materials']: mesh.materials.append(make_material(record))
    for polygon,row in zip(mesh.polygons,data['triangle_materials']): polygon.material_index=row['material']
    for attribute,width,default in [('displayColor',3,[.18,.18,.18]),('displayOpacity',1,1.)]:
        if not any(attribute in row for row in data['triangle_materials']):continue
        layer=mesh.color_attributes.new(name=attribute,type='FLOAT_COLOR',domain='CORNER')
        for polygon,row in zip(mesh.polygons,data['triangle_materials']):
            values=row.get(attribute,[default]*3)
            for loop_index,value in zip(polygon.loop_indices,values):
                layer.data[loop_index].color=(*value,1)if width==3 else(value,value,value,1)
    for name in ('diffuseColor','roughness','metallic','opacity','normal','normal_frame','emissiveColor'):
        if not any(name in row['uv'] for row in data['triangle_materials']): continue
        layer=mesh.uv_layers.new(name=name)
        for polygon,row in zip(mesh.polygons,data['triangle_materials']):
            coordinates=row['uv'].get(name,[[0,0]]*3)
            for loop_index,uv in zip(polygon.loop_indices,coordinates): layer.data[loop_index].uv=uv
    custom=None
    if any(data['normal_valid']):
        custom=[]
        for polygon,valid,normals in zip(mesh.polygons,data['normal_valid'],data['corner_normals']):
            polygon.use_smooth=True
            custom.extend(normals if valid else [list(polygon.normal)]*3)
        mesh.normals_split_custom_set(custom)
    # Bpy owns the complete copied mesh, corner attributes and custom normals.
    # Release decoded exchange lists before Cycles allocates its acceleration data.
    del data,points,custom
    import gc
    gc.collect()
    opaque=bpy.data.materials.new('OpaqueCoverage'); opaque.use_nodes=True
    nodes=opaque.node_tree.nodes; nodes.clear(); emit=nodes.new('ShaderNodeEmission')
    emit.inputs[0].default_value=(1,1,1,1); emit.inputs[1].default_value=1
    surface=nodes.new('ShaderNodeOutputMaterial'); opaque.node_tree.links.new(emit.outputs[0],surface.inputs[0])
    for view in rig['views']:
        camera_data=bpy.data.cameras.new(view['id']); camera=bpy.data.objects.new(view['id'],camera_data)
        scene.collection.objects.link(camera); camera.matrix_world=Matrix(view['camera_to_world'])
        camera_data.clip_start=view['clip_start']; camera_data.clip_end=view['clip_end']
        if view['projection']=='orthographic':
            camera_data.type='ORTHO'
            # Blender ortho_scale is width at landscape aspect, height at
            # portrait; our manifest always specifies vertical visible span.
            camera_data.ortho_scale=view['orthographic_scale']*max(1,rig['resolution'][0]/rig['resolution'][1])
        else:
            camera_data.type='PERSP'; camera_data.lens=view['focal_length_mm']
            camera_data.sensor_width=view['sensor_width_mm']; camera_data.sensor_fit='HORIZONTAL'
        scene.camera=camera
        scene.view_layers[0].material_override=None
        background.inputs['Color'].default_value=(*recipe['background_linear_rgb'],1)
        background.inputs['Strength'].default_value=recipe['world_strength']
        scene.render.filepath=str(output/(view['id']+'.png')); bpy.ops.render.render(write_still=True)
        scene.view_layers[0].material_override=opaque
        background.inputs['Color'].default_value=(0,0,0,1); background.inputs['Strength'].default_value=0
        scene.render.filepath=str(output/(view['id']+'-mask.png')); bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera,do_unlink=True)
    (output/'worker-receipt.json').write_text(json.dumps({'renderer':{'name':'Blender','version':bpy.app.version_string,
        'engine':'Cycles','device':scene.cycles.device,'threads':scene.render.threads,'samples':scene.cycles.samples,'recipe':recipe}},sort_keys=True))


if __name__=='__main__': main()
