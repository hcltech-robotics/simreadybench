"""Independent OpenUSD → neutral scene manifest → public Blender/Cycles CPU.

Camera contract ``simready-cameras-v1``: exactly four uniquely named views,
``resolution: [width,height]``, and a separate ``observation`` camera object.
``observed_camera_id`` names that fifth camera; legacy four-camera rigs may name
one comparison view instead. Release preparation requires the fifth camera.
``camera_to_world`` stores matrix *rows*, acting on homogeneous column vectors
in metres/Z-up. Columns are camera right, up, backward, and position; viewing
is along local -Z. Orthographic views specify vertical ``orthographic_scale``;
perspective views specify ``focal_length_mm`` and ``sensor_width_mm`` (horizontal
sensor fit). Every view supplies positive ``clip_start`` < ``clip_end``.
No camera fitting, image registration, or scale fitting takes place.

Only universal UsdPreviewSurface colour/roughness/metallic/opacity and supported
UsdUVTexture/PrimvarReader networks are qualified. Unsupported source features
return typed unknown receipts before rendering, never placeholder materials.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image

LEGACY_RECIPE = {
    'schema': 'simready-neutral-cycles-v1', 'blender_version':'4.3.2', 'engine': 'CYCLES', 'device': 'CPU',
    'texture_input_precision':'lossless streamed 8-bit RGBA canonical PNG; original 16-bit PNG samples preserved for native Blender decoding',
    'maximum_triangles':4_000_000,'maximum_surface_bytes':1024**3,'maximum_decoded_texture_bytes':1024**3,
    'samples': 8, 'seed': 1729, 'adaptive_sampling': False, 'denoising': False,
    'view_transform': 'Standard', 'look': 'None', 'exposure': 0.0, 'gamma': 1.0,
    'background_linear_rgb': [0.05, 0.05, 0.05], 'world_strength': 0.8,
    'lights': [{'position':[4,-4,6], 'energy':450., 'size':5.},
               {'position':[-4,-1,3], 'energy':250., 'size':4.},
               {'position':[0,5,4], 'energy':350., 'size':4.}],
    'film_transparent': False, 'pixel_filter_width': 1.5, 'dither_intensity':0.0,
    'mask': 'opaque-white-emission-override-black-world-nonzero-coverage',
    'alpha': 'PreviewSurface opacity via transparent/surface mix; mask ignores opacity',
    'zero_authored_normal_policy':'preserve zero vectors for Blender auto normals; normalize finite nonzero authored normals; nonfinite normals are unsupported',
    'unbound_material_policy':'authored displayColor/displayOpacity per corner; otherwise renderer defaults linear RGB0.18, roughness0.5, metallic0, opacity1; source material remains unbound and diagnostic N/A',
}

# Keep the published eight-sample recipe byte-for-byte identifiable. The new
# default changes only sample count and the explicitly frozen CPU thread count.
RECIPE = {**json.loads(json.dumps(LEGACY_RECIPE)), 'samples':256, 'threads':4}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)
    digest=hashlib.sha256()
    # A large scene must not allocate a second multi-gigabyte UTF-8 buffer.
    for start in range(0,len(encoded),1024**2):
        digest.update(encoded[start:start+1024**2].encode())
    return digest.hexdigest()


def validate_recipe(recipe=None):
    """Copy one exact supported CPU profile; arbitrary rendering changes fail.

    The default is CPU256/four threads. Explicit LEGACY_RECIPE retains the
    published CPU8/one-thread settings for characterization; fixture eligibility
    is controlled separately by the evaluator's frozen qualification contract.
    """
    selected=RECIPE if recipe is None else recipe
    try:
        if not isinstance(selected,dict):raise ValueError('Recipe must be an object')
        selected=json.loads(json.dumps(selected,allow_nan=False))
        if _identity(selected) not in {_identity(RECIPE),_identity(LEGACY_RECIPE)}:
            raise ValueError('Unrecognized frozen profile')
    except (TypeError,ValueError) as exc:
        raise ValueError('An exact qualified CPU recipe is required') from exc
    return selected


def _write_scene_json(path,value):
    encoded=json.dumps(value,allow_nan=False)
    with Path(path).open('w',encoding='utf-8',newline='')as stream:
        for start in range(0,len(encoded),1024**2):
            stream.write(encoded[start:start+1024**2])


def validate_cameras(manifest):
    """Validate and copy a frozen camera manifest; raise on ambiguous cameras."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    if manifest.get('schema') != 'simready-cameras-v1':
        raise ValueError('Expected simready-cameras-v1')
    size = manifest.get('resolution')
    if not isinstance(size, list) or len(size) != 2 or any(
        type(v) is not int or not 11 <= v <= 4096 for v in size):
        raise ValueError('resolution requires two integers in [11,4096]')
    views = manifest.get('views', [])
    if len(views) != 4:
        raise ValueError('Exactly four camera views are required')
    observation=manifest.get('observation')
    if observation is not None and not isinstance(observation,dict):
        raise ValueError('observation must be a camera object')
    all_views=views+([observation] if observation is not None else [])
    ids = []
    for view in all_views:
        name = view.get('id')
        if not isinstance(name, str) or not name or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Camera id must be a safe filename identifier')
        ids.append(name)
        matrix = np.asarray(view.get('camera_to_world'), dtype=float)
        if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(
                matrix[3], [0,0,0,1], atol=1e-9):
            raise ValueError('Invalid camera matrix')
        rotation = matrix[:3,:3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
                np.linalg.det(rotation), 1, atol=1e-7):
            raise ValueError('Camera matrix must be rigid and right-handed')
        near, far = float(view.get('clip_start', 0)), float(view.get('clip_end', 0))
        if not math.isfinite(near+far) or not 0 < near < far:
            raise ValueError('Positive ordered clipping planes required')
        if view.get('projection') == 'orthographic':
            fields = ['orthographic_scale']
        elif view.get('projection') == 'perspective':
            fields = ['focal_length_mm', 'sensor_width_mm']
        else:
            raise ValueError('Unsupported camera projection')
        if any(not math.isfinite(float(view.get(k, 0))) or float(view.get(k, 0)) <= 0 for k in fields):
            raise ValueError('Positive finite projection parameters required')
    if len(set(ids)) != len(all_views) or manifest.get('observed_camera_id') not in ids:
        raise ValueError('Unique ids and one explicitly observed camera required')
    if observation is not None:
        if manifest['observed_camera_id']!=observation['id']:
            raise ValueError('observed_camera_id must name the separate observation')
        matrices=[np.asarray(view['camera_to_world'],float) for view in all_views]
        for i,matrix in enumerate(matrices):
            if any(np.allclose(matrix,other,atol=1e-9,rtol=0) for other in matrices[i+1:]):
                raise ValueError('All five cameras must have distinct camera matrices')
    return manifest


def _alignment_matrix(alignment):
    if alignment is None:
        return np.eye(4)
    if isinstance(alignment, dict):
        if 'matrix' in alignment:
            alignment = alignment['matrix']
        elif 'rotation' in alignment and 'translation_m' in alignment:
            value = np.eye(4); value[:3,:3] = alignment['rotation']
            value[:3,3] = alignment['translation_m']; alignment = value
        else:
            raise ValueError('Alignment requires matrix or rotation/translation_m')
    matrix = np.asarray(alignment, float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0,0,0,1]):
        raise ValueError('Invalid alignment matrix')
    r = matrix[:3,:3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r),1):
        raise ValueError('Alignment must preserve scale')
    if not np.allclose(r[2], [0,0,1], atol=1e-7):
        raise ValueError('Alignment must be one whole-object Z yaw and translation')
    return matrix


class _TextureFile:
    """Immutable content-addressed texture metadata; never owns decoded RGBA."""
    __slots__=('file','digest','shape','bit_depth')

    def __init__(self,file,digest,shape,bit_depth):
        self.file=Path(file);self.digest=digest;self.shape=tuple(shape);self.bit_depth=bit_depth


class _TextureBridge:
    """Read the public PreviewSurface texture graph independently of M diagnostics.

    UsdTransform2d applies scale, then counterclockwise rotation in degrees,
    then translation, per the OpenUSD PreviewSurface specification.
    """
    def __init__(self, source,package_root=None,texture_dir=None):
        source=Path(source).resolve()
        self.root=Path(package_root).resolve()if package_root is not None else source.parent
        if not self.root.is_dir()or not source.is_relative_to(self.root):
            raise ValueError('Source must be contained in an existing package root')
        self._temporary_textures=tempfile.TemporaryDirectory(prefix='simready-textures-')if texture_dir is None else None
        self.texture_dir=Path(self._temporary_textures.name)if texture_dir is None else Path(texture_dir)
        self.texture_dir.mkdir(parents=True,exist_ok=True)
        self.inputs={};self.images={};self.primvars={}
        self.encoded_images={};self.image_contents={};self.image_metadata={};self.peak_decoded_rgba_bytes=0
        self.current_metadata=None;self.current_prim=None;self.current_raw_uv={};self.current_uv={}
        self.constant_uv=np.zeros((3,2))

    @staticmethod
    def _literal(shader,name,default):
        inp=shader.GetInput(name)
        if not inp: return default
        if inp.HasConnectedSource(): raise ValueError('Connected parameter unsupported: '+name)
        value=inp.Get()
        return default if value is None else value

    def _st(self,inp,seen=()):
        from pxr import UsdShade
        if not inp or not inp.HasConnectedSource(): raise ValueError('Texture st needs a connected primvar reader')
        source=inp.GetConnectedSource();node=UsdShade.Shader(source[0].GetPrim());path=str(node.GetPath())
        if path in seen or len(seen)>=8: raise ValueError('Cyclic/deep texture coordinate graph')
        identifier=node.GetIdAttr().Get()
        if identifier=='UsdPrimvarReader_float2':
            name=str(self._literal(node,'varname',''))
            if not name: raise ValueError('Missing texture coordinate primvar')
            return name,np.eye(3)
        if identifier!='UsdTransform2d': raise ValueError('Unsupported texture coordinate node: '+str(identifier))
        name,previous=self._st(node.GetInput('in'),(*seen,path))
        scale=np.asarray(self._literal(node,'scale',[1,1]),float)
        translation=np.asarray(self._literal(node,'translation',[0,0]),float)
        angle=math.radians(float(self._literal(node,'rotation',0)))
        if scale.shape!=(2,) or translation.shape!=(2,) or not np.isfinite([*scale,*translation,angle]).all():
            raise ValueError('Invalid UsdTransform2d parameters')
        c,s=math.cos(angle),math.sin(angle)
        transform=np.eye(3);transform[:2,:2]=np.array([[c,-s],[s,c]])@np.diag(scale)
        transform[:2,2]=translation
        return name,transform@previous

    def _read_image(self,file):
        file=Path(file).resolve()
        if not file.is_relative_to(self.root):raise ValueError('Texture escapes source package')
        if file in self.images:return self.images[file]
        size=file.stat().st_size
        if size>128*1024**2:raise MemoryError('Encoded texture exceeds resource bound')
        encoded_digest=hashlib.sha256()
        with file.open('rb')as stream:
            header=stream.read(33);encoded_digest.update(header)
            while block:=stream.read(1024**2):encoded_digest.update(block)
        encoded_digest=encoded_digest.hexdigest()
        png_depth=header[24]if header[:8]==b'\x89PNG\r\n\x1a\n'and header[12:16]==b'IHDR'else None
        if png_depth is not None:
            width,height=struct.unpack('>II',header[16:24])
            if width*height>16*1024**2:raise MemoryError('Decoded texture exceeds pixel resource bound')
        with Image.open(file)as image:
            width,height=image.size;mode=image.mode
            if width*height>16*1024**2:raise MemoryError('Decoded texture exceeds pixel resource bound')
            native=png_depth==16
            if not native and(mode in('I','F')or mode.startswith('I;16')):
                raise ValueError('High-precision texture requires supported native PNG input')
            bit_depth=16 if native else 8
            rgba_bytes=width*height*4*(bit_depth//8)
            peak_pixel_bytes=0 if native else width*height*(len(image.getbands())+8)
            if max(rgba_bytes,peak_pixel_bytes)>RECIPE['maximum_decoded_texture_bytes']:
                raise MemoryError('Single decoded texture exceeds peak allocation bound')
            if encoded_digest in self.encoded_images:
                result=self.encoded_images[encoded_digest]
            elif native:
                # Keep every original 16-bit sample and let the qualified public
                # Blender decoder consume it. Pillow's RGB/RGBA view is only 8-bit.
                key='native16:'+encoded_digest
                result=self.image_contents.get(key)
                if result is None:
                    target=self.texture_dir/('native16-'+encoded_digest+'.png')
                    if not target.exists():shutil.copyfile(file,target)
                    result=_TextureFile(target,key,(height,width,4),16)
                    self.image_contents[key]=result
            else:
                # Bound original decoder pixels, one RGBA image and its immutable
                # byte view. No float array or per-material decoded copy is held.
                rgba=image.convert('RGBA');rgba.info.clear()
                pixels=rgba.tobytes();shape=(height,width,4)
                self.peak_decoded_rgba_bytes=max(self.peak_decoded_rgba_bytes,peak_pixel_bytes)
                digest=hashlib.sha256();digest.update(pixels);digest.update(str(shape).encode())
                key=digest.hexdigest();result=self.image_contents.get(key)
                if result is None:
                    target=self.texture_dir/(key+'.png')
                    if not target.exists():rgba.save(target)
                    result=_TextureFile(target,key,shape,8)
                    self.image_contents[key]=result
                del pixels,rgba
        self.encoded_images[encoded_digest]=result
        self.images[file]=result
        self.image_metadata[file]={'width':width,'height':height,'bit_depth':bit_depth,'pillow_mode':mode,
                                  'encoded_sha256':encoded_digest,'source_sample_rgba_bytes':rgba_bytes,
                                  'native_precision_preserved':native}
        return result

    def texture_statistics(self):
        return {'source_paths':len(self.images),'unique_texture_contents':len(self.image_contents),
                'retained_rgba_bytes':0,'peak_decoded_rgba_bytes':self.peak_decoded_rgba_bytes,
                'maximum_peak_decoded_rgba_bytes':RECIPE['maximum_decoded_texture_bytes'],
                'source_sample_rgba_bytes':sum(item['source_sample_rgba_bytes']for item in self.image_metadata.values()),
                'files':[{'path':str(file.relative_to(self.root)),**metadata}for file,metadata in self.image_metadata.items()],
                'scope':'bridge streams one lossless texture at a time; Blender native allocations are bounded and measured separately'}

    def _compile(self,inp):
        from pxr import UsdShade
        source=inp.GetConnectedSource();node=UsdShade.Shader(source[0].GetPrim())
        if node.GetIdAttr().Get()!='UsdUVTexture':raise ValueError('Unsupported visual shader/texture input: '+inp.GetBaseName())
        channel=str(source[1]);channels={'rgb':[0,1,2],'r':[0],'g':[1],'b':[2],'a':[3]}.get(channel)
        if channels is None:raise ValueError('Unsupported UVTexture output channel')
        space=str(self._literal(node,'sourceColorSpace','auto'))
        if space not in ('raw','sRGB','auto'):raise ValueError('Unsupported texture colour space: '+space)
        wrap=[str(self._literal(node,'wrap'+axis,'repeat'))for axis in ('S','T')]
        if any(value not in ('repeat','clamp')for value in wrap):raise ValueError('Unsupported texture wrap mode')
        scale=np.asarray(self._literal(node,'scale',[1]*4),float);bias=np.asarray(self._literal(node,'bias',[0]*4),float)
        fallback=np.asarray(self._literal(node,'fallback',[0,0,0,1]),float)
        if any(value.shape!=(4,)for value in (scale,bias,fallback))or not np.isfinite([*scale,*bias,*fallback]).all():
            raise ValueError('Invalid texture scale/bias/fallback')
        asset=self._literal(node,'file',None);authored=getattr(asset,'path','');resolved=getattr(asset,'resolvedPath','')
        descriptor={'channels':channels,'wrap':wrap,'sRGB':space=='sRGB' or space=='auto' and Path(authored).suffix.lower()in('.png','.jpg','.jpeg'),
                    'scale':scale[channels],'bias':bias[channels],'fallback':fallback.tolist()}
        if not authored:
            descriptor.update(fallback_only=True,sRGB=False)
        elif UsdShade.UdimUtils.IsUdimIdentifier(authored):
            specs=node.GetInput('file').GetAttr().GetPropertyStack()
            layer=specs[0].layer if specs else None
            tiles=UsdShade.UdimUtils.ResolveUdimTilePaths(authored,layer)
            if not tiles:raise ValueError('UDIM template has no resolvable tiles')
            if len(tiles)>100:raise MemoryError('UDIM tile resource bound exceeded')
            descriptor.update(udim=True,textures={str(tile):self._read_image(file)for file,tile in tiles})
        else:
            if not resolved:raise ValueError('Unresolved authored UVTexture file')
            descriptor['texture']=self._read_image(resolved)
        if descriptor.get('fallback_only'):
            name,transform=None,np.eye(3)
        else:name,transform=self._st(node.GetInput('st'))
        return descriptor,name,transform

    def display(self,prim,name,metadata,indices,default):
        from pxr import UsdGeom
        key=(str(prim.GetPath()),name)
        if key not in self.primvars:
            pv=UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
            data=pv.ComputeFlattened() if pv else None
            self.primvars[key]=(np.asarray(data,float),str(pv.GetInterpolation())) if data is not None and len(data)else(None,'constant')
        values,interpolation=self.primvars[key]
        if values is None:return np.repeat(np.asarray(default,float)[None],3,axis=0),'renderer_default'
        face=int(metadata[0]);corners=np.asarray(metadata[1:],int)
        if interpolation=='faceVarying':result=values[corners]
        elif interpolation in('vertex','varying'):result=values[np.asarray(indices)[corners]]
        elif interpolation=='uniform':result=np.repeat(values[face][None],3,axis=0)
        elif interpolation=='constant':result=np.repeat(values[0][None],3,axis=0)
        else:raise ValueError('Unsupported display primvar interpolation')
        if not np.isfinite(result).all():raise ValueError('Nonfinite display primvar')
        return result,'authored_display_primvar'

    def value(self,shader,name,default,prim,metadata,indices):
        from pxr import UsdGeom
        key=(str(shader.GetPath()),name)
        if key not in self.inputs:
            inp=shader.GetInput(name)
            if not inp or not inp.HasConnectedSource():
                value=inp.Get() if inp else None
                value=default if value is None else value
                value=np.asarray(value,float)
                if not np.isfinite(value).all(): raise ValueError('Nonfinite visual shader input: '+name)
                self.inputs[key]=float(value) if value.ndim==0 else value.tolist()
            else: self.inputs[key]=self._compile(inp)
        value=self.inputs[key]
        if not isinstance(value,tuple): return value
        descriptor,varname,transform=value
        if varname is None:
            return {**descriptor,'uv':self.constant_uv,'frame_uv':self.constant_uv}
        # Only the current original-corner row is cached. Material channels
        # sharing a primvar/transform reuse the exact same finite UV values.
        if self.current_metadata is not metadata or self.current_prim!=prim:
            self.current_metadata=metadata;self.current_prim=prim
            self.current_raw_uv={};self.current_uv={}
        transformed_key=(varname,transform.tobytes())
        if transformed_key not in self.current_uv:
            if varname not in self.current_raw_uv:
                uvkey=(str(prim.GetPath()),varname)
                if uvkey not in self.primvars:
                    pv=UsdGeom.PrimvarsAPI(prim).GetPrimvar(varname)
                    if not pv:raise ValueError('Missing UV primvar '+varname)
                    self.primvars[uvkey]=(np.asarray(pv.ComputeFlattened(),float),str(pv.GetInterpolation()))
                values,interpolation=self.primvars[uvkey];face=int(metadata[0]);corners=np.asarray(metadata[1:],int)
                if interpolation=='faceVarying':uv=values[corners]
                elif interpolation in('vertex','varying'):uv=values[np.asarray(indices)[corners]]
                elif interpolation=='uniform':uv=np.repeat(values[face][None],3,axis=0)
                elif interpolation=='constant':uv=np.repeat(values[0][None],3,axis=0)
                else:raise ValueError('Unsupported UV interpolation: '+interpolation)
                if uv.shape!=(3,2)or not np.isfinite(uv).all():raise ValueError('Invalid UV coordinates')
                self.current_raw_uv[varname]=uv
            uv=self.current_raw_uv[varname]
            self.current_uv[transformed_key]=uv@transform[:2,:2].T+transform[:2,2]
        return {**descriptor,'uv':self.current_uv[transformed_key],'frame_uv':self.current_raw_uv[varname]}



def _extract_scene(source, target, alignment,package_root=None):
    from pxr import UsdShade, UsdGeom, Usd
    from .usd import observe
    observation = observe(source, maximum_triangles=RECIPE['maximum_triangles'],maximum_surface_bytes=RECIPE['maximum_surface_bytes'])
    triangles = np.asarray(observation['triangles'], float)
    triangles = triangles @ alignment[:3,:3].T + alignment[:3,3]
    materials = observation['materials']
    rows, material_rows, lookup, unsupported = [], [], {}, []
    texture_dir=target/'textures';texture_dir.mkdir(parents=True,exist_ok=True)
    diagnostics=[];issues=set();bridge=_TextureBridge(source,package_root,texture_dir=texture_dir)
    triangle_context=[None]*len(triangles)
    material_status={'bound_preview_surface_triangles':0,'unbound_triangles':0}
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    double_sided=np.zeros(len(triangles),bool)
    reversed_winding=np.zeros(len(triangles),bool)
    used_shaders=set()
    for start,end,prim,indices,metadata in materials.groups:
        gprim=UsdGeom.Gprim(prim)
        double_sided[start:end]=bool(gprim.GetDoubleSidedAttr().Get())
        determinant=np.linalg.det(np.asarray(xforms.GetLocalToWorldTransform(prim),float)[:3,:3])
        reversed_winding[start:end]=(gprim.GetOrientationAttr().Get()=='leftHanded') != (determinant<0)
        subsets=UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
        face_targets={}
        for subset in subsets:
            for face in subset.GetIndicesAttr().Get() or []:face_targets.setdefault(int(face),subset.GetPrim())
        bindings={}
        for offset,meta in enumerate(metadata):
            bound=face_targets.get(int(meta[0]),prim); key=str(bound.GetPath())
            if key not in bindings:
                material,_=UsdShade.MaterialBindingAPI(bound).ComputeBoundMaterial()
                bindings[key]=(material.ComputeSurfaceSource()[0],True) if material else (None,False)
            triangle_context[start+offset]=(*bindings[key],prim,meta,indices)
        targets=[prim]+[subset.GetPrim() for subset in subsets]
        for bound in targets:
            material,_=UsdShade.MaterialBindingAPI(bound).ComputeBoundMaterial()
            if not material: continue
            shader=material.ComputeSurfaceSource()[0]
            if shader: used_shaders.add(str(shader.GetPath()))
            if material.GetDisplacementOutput() and material.GetDisplacementOutput().HasConnectedSource():
                unsupported.append({'prim':str(bound.GetPath()),'reason':'Unsupported displacement output'})
    # Geometry and material observation is lazy; visit every triangle before
    # assessing support. Each corner's UV survives triangulation in this API.
    udim_files={};material_templates={}
    fields=[('diffuseColor',[.18,.18,.18]),('roughness',.5),('metallic',0.),('opacity',1.),
            ('normal',[0.,0.,1.]),('emissiveColor',[0.,0.,0.]),('ior',1.5)]
    for index in range(len(triangles)):
        shader,bound,prim,metadata,indices=triangle_context[index]
        if bound and (not shader or shader.GetIdAttr().Get()!='UsdPreviewSurface'):
            issue=(str(prim.GetPath()),'No supported bound universal UsdPreviewSurface material')
            if issue not in issues:unsupported.append({'prim':issue[0],'reason':issue[1]});issues.add(issue)
            continue
        record, uv = {}, {}
        display={};uv_lists={}
        def corner_list(value):
            array=np.asarray(value,float);key=array.tobytes()
            if key not in uv_lists:uv_lists[key]=array.tolist()
            return uv_lists[key]
        if not bound:
            material_status['unbound_triangles']+=1
            record.update(source_material_status='unbound',material_diagnostic_status='not_applicable')
            for display_name,default in [('displayColor',[.18,.18,.18]),('displayOpacity',1.)]:
                values,provenance=bridge.display(prim,display_name,metadata,indices,default)
                display[display_name]=values.tolist()
                issue=(str(prim.GetPath()),display_name,provenance)
                if issue not in issues:
                    diagnostics.append({'prim':issue[0],'reason':'Unbound '+display_name+': '+provenance});issues.add(issue)
        else:material_status['bound_preview_surface_triangles']+=1
        template_key=(str(shader.GetPath())if bound else None,bool(double_sided[index]))
        cached=material_templates.get(template_key)
        selected_fields=cached[1]if cached is not None else fields
        for name, default in selected_fields:
            try:
                value=bridge.value(shader,name,default,prim,metadata,indices) if bound else default
                if name=='ior' and (not isinstance(value,(int,float)) or value<=0):
                    raise ValueError('IOR requires a positive literal scalar')
            except (OSError,ValueError,RuntimeError,IndexError,TypeError) as exc:
                issue=(str(shader.GetPath()),name,str(exc))
                if issue not in issues:
                    unsupported.append({'prim':issue[0],'reason':'Unsupported PreviewSurface input '+name+': '+str(exc)});issues.add(issue)
                value=default
            if isinstance(value, dict):
                uv[name]=corner_list(value['uv'])
                if name=='normal':uv['normal_frame']=corner_list(value['frame_uv'])
                if cached is not None:continue
                record[name]={key:np.asarray(value[key]).tolist()if key in('channels','scale','bias')else value[key]
                              for key in('channels','scale','bias','wrap','sRGB','fallback')}
                if value.get('fallback_only'):
                    record[name]['fallback_only']=True
                    issue=(str(shader.GetPath()),name,'UsdUVTexture file absent: authored/default standard fallback')
                    if issue not in issues:
                        diagnostics.append({'prim':issue[0],'reason':issue[2],'input':name});issues.add(issue)
                elif value.get('udim'):
                    cache_key=id(value['textures'])
                    if cache_key not in udim_files:
                        tile_hashes={tile:texture.digest for tile,texture in value['textures'].items()}
                        prefix=_identity(tile_hashes)
                        tile_files={tile:str((texture_dir/(prefix+'.'+tile+'.png')).resolve())for tile in tile_hashes}
                        for tile,file in tile_files.items():
                            if not Path(file).exists():shutil.copyfile(value['textures'][tile].file,file)
                        udim_files[cache_key]=(str((texture_dir/(prefix+'.<UDIM>.png')).resolve()),tile_files)
                    file,tiles=udim_files[cache_key]
                    record[name].update(udim=True,file=file,tiles=tiles)
                else:
                    record[name]['file']=str(value['texture'].file.resolve())

            elif cached is None:
                record[name] = value
                if name=='normal' and not np.allclose(value,[0,0,1]):
                    unsupported.append({'prim':str(shader.GetPath()),'reason':'Constant nondefault tangent normal lacks a qualified tangent frame'})
        record['double_sided']=bool(double_sided[index])
        if reversed_winding[index]:
            triangles[index]=triangles[index][[0,2,1]]
            uv={name:corner_list(np.asarray(value)[[0,2,1]])for name,value in uv.items()}
            display={name:np.asarray(value)[[0,2,1]].tolist()for name,value in display.items()}
        if cached is None:
            key=_identity(record)
            if key not in lookup:
                lookup[key]=len(material_rows);material_rows.append(record)
            material_index=lookup[key]
            # Values/graphs are default-time constants per effective shader;
            # every triangle still samples and validates its own UV corners.
            uv_fields=[(name,default)for name,default in fields if isinstance(record.get(name),dict)]
            material_templates[template_key]=(material_index,uv_fields)
        else:material_index=cached[0]
        rows.append({'material':material_index,'uv':uv,**display})
    unsupported.extend(observation['eligibility'].get('unsupported', []))
    # Refuse unsupported authored shading features rather than quietly losing
    # them while translating the qualified base PreviewSurface subset.
    for prim in materials.stage.Traverse(Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()) not in used_shaders: continue
        if not prim.IsA(UsdShade.Shader): continue
        shader=UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != 'UsdPreviewSurface': continue
        allowed={'diffuseColor','roughness','metallic','opacity','normal','emissiveColor','ior'}
        benign={'useSpecularWorkflow':0,'ior':1.5,'opacityThreshold':0.,
                'clearcoat':0.,'clearcoatRoughness':.01,'emissiveColor':(0.,0.,0.),
                'occlusion':1.,'displacement':0.,'normal':(0.,0.,1.)}
        for inp in shader.GetInputs():
            name=inp.GetBaseName()
            if name in allowed: continue
            value=inp.Get()
            coat=shader.GetInput('clearcoat')
            coat_inactive=not coat or not coat.HasConnectedSource() and coat.Get() in (None,0)
            if name=='specular':
                diagnostics.append({'prim':str(prim.GetPath()),'reason':'Unused nonstandard UsdPreviewSurface input: specular'})
                continue
            if name=='clearcoatRoughness' and coat_inactive:
                diagnostics.append({'prim':str(prim.GetPath()),'reason':'Inactive clearcoatRoughness because effective clearcoat is zero'})
                continue
            if inp.HasConnectedSource() or (value is not None and
                    (name not in benign or not np.allclose(value,benign[name]))):
                unsupported.append({'prim':str(prim.GetPath()),'reason':'Unsupported PreviewSurface input: '+name})
    # Authored normals are interpolated from original USD face corners. No
    # normal data is reconstructed from UVs or merged across seams.
    normals = np.zeros_like(triangles); normal_valid = np.zeros(len(triangles),bool)
    from pxr import Usd
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    up=UsdGeom.GetStageUpAxis(materials.stage)
    canonical=np.array([[1,0,0],[0,0,-1],[0,1,0]]) if up=='Y' else np.eye(3)
    for start,end,prim,indices,metadata in materials.groups:
        mesh=UsdGeom.Mesh(prim)
        if not mesh: continue
        raw=mesh.GetNormalsAttr().Get()
        if raw is None or len(raw)==0: continue
        raw=np.asarray(raw,float); interp=str(mesh.GetNormalsInterpolation())
        transform=np.asarray(xforms.GetLocalToWorldTransform(prim),float)[:3,:3]
        normal_transform=alignment[:3,:3] @ canonical @ np.linalg.inv(transform)
        for offset,meta in enumerate(metadata):
            face=int(meta[0]); corners=np.asarray(meta[1:],int)
            if interp=='faceVarying': n=raw[corners]
            elif interp in ('vertex','varying'): n=raw[np.asarray(indices)[corners]]
            elif interp=='uniform': n=np.repeat(raw[face][None],3,axis=0)
            elif interp=='constant': n=np.repeat(raw[0][None],3,axis=0)
            else:
                unsupported.append({'prim':str(prim.GetPath()),'reason':'Unsupported normal interpolation'}); break
            n=n @ normal_transform.T
            lengths=np.linalg.norm(n,axis=1)
            if not np.isfinite(n).all():
                unsupported.append({'prim':str(prim.GetPath()),'reason':'Nonfinite authored normals'});break
            zero=lengths==0
            if np.any(zero):
                issue=(str(prim.GetPath()),'zero_authored_normals')
                if issue not in issues:
                    diagnostic={'prim':issue[0],'reason':'Zero-length authored normal corners preserved for Blender auto normals','corner_count':0}
                    diagnostics.append(diagnostic);issues.add(issue)
                else:diagnostic=next(row for row in diagnostics if row.get('prim')==issue[0]and row.get('reason','').startswith('Zero-length authored normal'))
                diagnostic['corner_count']+=int(zero.sum())
            n=np.divide(n,lengths[:,None],out=np.zeros_like(n),where=lengths[:,None]>0)
            if reversed_winding[start+offset]: n=n[[0,2,1]]
            normals[start+offset]=n; normal_valid[start+offset]=True
    # Harmless zero-area omitted surfaces are explicitly reported by the USD
    # observer but do not require a replacement surface or shader.
    accepted_prefixes=('Zero-area','Nonplanar polygon uses deterministic original-corner fan triangulation','Repeated-position quad uses original-corner fan')
    diagnostics.extend(u for u in unsupported if u.get('reason','').startswith(accepted_prefixes))
    unsupported=[u for u in unsupported if not u.get('reason','').startswith(accepted_prefixes)]
    scene={'schema':'simready-neutral-scene-v1','triangles':triangles.tolist(),
           'triangle_materials':rows,'materials':material_rows,
           'corner_normals':normals.tolist(),'normal_valid':normal_valid.tolist()}
    return scene, {'status':'supported' if not unsupported else 'unknown',
                   'unsupported':unsupported,'diagnostics':diagnostics,'source_material_status':material_status,'unbound_policy':RECIPE['unbound_material_policy'],'units':'metres/Z-up',
                   'triangles':len(triangles),'texture_cache':bridge.texture_statistics(),'surface_scope':observation['eligibility'],
                   'shader_scope':'universal PreviewSurface colour, roughness, metallic, opacity, tangent normal, emission, literal IOR; UsdUVTexture including UDIM and fallback + PrimvarReader + UsdTransform2d; declared unbound display/default policy'}


def render_asset(source, cameras_manifest, output_dir, alignment=None, timeout=180, blender='blender',role='comparison',package_root=None,recipe=None):
    """Render four exact frozen cameras; unavailable/support failures stay unknown.

    ``alignment`` may be a 4x4 matrix or geometry alignment receipt containing
    ``rotation`` and ``translation_m``. Receipt paths point to persisted PNGs.
    Source packages must have passed the caller's immutable package boundary.
    ``recipe`` selects an exact supported CPU profile; None uses CPU256/four
    threads. Explicit LEGACY_RECIPE preserves CPU8/one-thread characterization.
    """
    recipe=validate_recipe(recipe)
    cameras=validate_cameras(cameras_manifest); matrix=_alignment_matrix(alignment)
    source=Path(source).resolve()
    package_root=Path(package_root).resolve()if package_root is not None else source.parent
    if not package_root.is_dir()or not source.is_relative_to(package_root):
        raise ValueError('Source must be contained in an existing package root')
    if role not in ('comparison','observation','all'):raise ValueError('Unknown render role')
    if role in ('observation','all') and 'observation' not in cameras:
        raise ValueError('Observation role requires a separate observation camera')
    selected_views=(cameras['views']+[cameras['observation']]if role=='all'else cameras['views']if role=='comparison'else[cameras['observation']])
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'simready-render-receipt-v1','status':'unknown','images':[],
             'camera_sha256':_identity(cameras),'recipe':recipe,'recipe_sha256':_identity(recipe),
             'alignment':matrix.tolist(),'variant':'raw' if alignment is None else 'globally_aligned','role':role}
    phase='extract'
    try:
        scene,support=_extract_scene(source,output,matrix,package_root=package_root)
        receipt['support']=support
        if support['status']!='supported':
            receipt['reason']='unsupported_scene'; return receipt
        if not scene['triangles']:
            receipt['reason']='empty_scene'; return receipt
        phase='scene_hash'
        scientific_scene={**scene,'materials':json.loads(json.dumps(scene['materials']))}
        for material in scientific_scene['materials']:
            for value in material.values():
                if isinstance(value,dict) and 'file' in value:
                    value['file']=Path(value['file']).name
                if isinstance(value,dict) and 'tiles' in value:
                    value['tiles']={tile:Path(file).name for tile,file in value['tiles'].items()}
        receipt['scene_sha256']=_identity(scientific_scene)
        worker_cameras={**cameras,'views':selected_views}
        scene.update(cameras=worker_cameras, recipe=recipe,output_dir=str(output))
        phase='exchange_write'
        scene_file=output/'scene.json';_write_scene_json(scene_file,scene)
        # The immutable exchange file now owns this data; retaining the parent
        # arrays while Blender parses them doubles peak memory for large meshes.
        del scientific_scene,scene
        import gc
        gc.collect()
        worker=Path(__file__).with_name('render_worker.py')
        executable=shutil.which(str(blender))
        if executable is None: raise OSError('Blender executable unavailable')
        receipt['renderer_binary_sha256']=_hash(executable)
        receipt['bridge_code_sha256']=_identity([_hash(__file__),_hash(worker)])
        phase='blender'
        process=subprocess.run([str(blender),'--background','--factory-startup','--threads',str(recipe.get('threads',1)),
                                '--python-exit-code','1','--python',str(worker),'--',str(scene_file)],
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               timeout=timeout,check=False)
        (output/'blender.log').write_bytes(process.stdout[-256*1024:])
        if process.returncode:
            receipt.update(reason='render_resource_bound'if b'\nMemoryError\n'in process.stdout else'renderer_failure',returncode=process.returncode,error_phase='blender');return receipt
        phase='images'
        worker_receipt=json.loads((output/'worker-receipt.json').read_text())
        receipt['renderer']=worker_receipt['renderer']
        for view in selected_views:
            file=output/(view['id']+'.png'); mask=output/(view['id']+'-mask.png')
            # Blender embeds dates/render durations even with stamps disabled.
            # Canonical PNG encoding retains pixels and removes volatile metadata.
            for artifact,mode in ((file,'RGB'),(mask,'L')):
                with Image.open(artifact) as image:
                    pixels=np.array(image.convert(mode))
                Image.fromarray(pixels).save(artifact,format='PNG',compress_level=9)
            with Image.open(mask) as image: foreground=int(np.count_nonzero(np.asarray(image.convert('L'))))
            receipt['images'].append({'camera_id':view['id'],'file':str(file),'mask_file':str(mask),
                                      'sha256':_hash(file),'mask_sha256':_hash(mask),
                                      'foreground_pixels':foreground})
        receipt.update(status='measured')
    except subprocess.TimeoutExpired:
        receipt['reason']='renderer_timeout'
    except MemoryError as exc:
        receipt.update(reason='render_resource_bound',error_type=type(exc).__name__,error=str(exc),error_phase=phase)
    except (OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
        receipt.update(reason='render_unavailable',error_type=type(exc).__name__,error=str(exc))
    return receipt


def _blur(values):
    # Separable normalized 11x11 Gaussian, reflect at image boundaries.
    axis=np.arange(-5,6,dtype=float); kernel=np.exp(-axis**2/(2*1.5**2)); kernel/=kernel.sum()
    result=values
    for dimension in (0,1):
        padding=[(0,0)]*result.ndim; padding[dimension]=(5,5)
        padded=np.pad(result,padding,mode='reflect')
        result=sum(weight*np.take(padded,range(i,i+result.shape[dimension]),axis=dimension)
                   for i,weight in enumerate(kernel))
    return result


def foreground_rgb_ssim(reference, candidate, mask):
    """Frozen sRGB local SSIM mean over opacity-independent foreground union."""
    x=np.asarray(reference,dtype=float)/255.; y=np.asarray(candidate,dtype=float)/255.
    mask=np.asarray(mask,bool)
    if x.shape!=y.shape or x.ndim!=3 or x.shape[2]!=3 or mask.shape!=x.shape[:2]:
        raise ValueError('RGB/mask dimensions do not match')
    if not mask.any(): raise ValueError('Empty foreground')
    ux,uy=_blur(x),_blur(y)
    vx=np.maximum(_blur(x*x)-ux*ux,0); vy=np.maximum(_blur(y*y)-uy*uy,0)
    covariance=_blur(x*y)-ux*uy
    value=((2*ux*uy+.01**2)*(2*covariance+.03**2)/
           ((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2)))
    return 100*float(np.clip(value[mask].mean(),0,1))


def _rows(images):
    if isinstance(images,dict):
        if images.get('status') not in (None,'measured'): return {}
        images=images.get('images',[])
    if not isinstance(images,list): return {}
    ids=[row.get('camera_id') for row in images]
    if len(set(ids))!=len(ids): return {}
    return {row['camera_id']:row for row in images}


def compare_views(reference_images, candidate_images, cameras_manifest):
    """Equal four-view pooling with fixed denominators and partial-known bounds."""
    cameras=validate_cameras(cameras_manifest)
    ref,cand=_rows(reference_images),_rows(candidate_images)
    for receipt,rows in ((reference_images,ref),(candidate_images,cand)):
        if isinstance(receipt,dict):
            if receipt.get('camera_sha256',_identity(cameras))!=_identity(cameras): rows.clear()
            if receipt.get('recipe_sha256',_identity(RECIPE))!=_identity(RECIPE): rows.clear()
    views=[]; total=0.; measured=0
    for camera in cameras['views']:
        name=camera['id']; row={'camera_id':name,'status':'unknown','score_percent':None,
                               'role':'observed' if name==cameras['observed_camera_id'] else 'novel'}
        try:
            r,c=ref[name],cand[name]
            for artifact in (r,c):
                for field,digest in (('file','sha256'),('mask_file','mask_sha256')):
                    if digest in artifact and _hash(artifact[field])!=artifact[digest]:
                        raise ValueError('Image receipt digest mismatch')
            def load(file,mode):
                with Image.open(file) as image:
                    if list(image.size)!=cameras['resolution']: raise ValueError('Frozen resolution mismatch')
                    return np.asarray(image.convert(mode))
            rm=load(r['mask_file'],'L')>0; cm=load(c['mask_file'],'L')>0
            if not rm.any(): raise ValueError('Empty frozen reference foreground')
            score=foreground_rgb_ssim(load(r['file'],'RGB'),load(c['file'],'RGB'),rm|cm)
            row.update(status='measured',score_percent=score,foreground_union_pixels=int((rm|cm).sum()),
                       reference_sha256=_hash(r['file']),candidate_sha256=_hash(c['file']),
                       reference_mask_sha256=_hash(r['mask_file']),candidate_mask_sha256=_hash(c['mask_file']))
            total+=score; measured+=1
        except (OSError,ValueError,KeyError,TypeError) as exc:
            row['reason']=str(exc)
        views.append(row)
    novel=[r['score_percent'] for r in views if r['role']=='novel' and r['status']=='measured']
    observed=next((r['score_percent'] for r in views if r['role']=='observed'),None)
    novel_eligible=sum(r['role']=='novel' for r in views)
    return {'schema':'simready-four-view-foreground-rgb-ssim-v1',
            'status':'measured' if measured==4 else 'unknown','score_percent':total/4 if measured==4 else None,
            'eligible':4,'measured':measured,'coverage_fraction':measured/4,
            'lower_bound_percent':total/4,'upper_bound_percent':(total+100*(4-measured))/4,
            'camera_sha256':_identity(cameras),'recipe_sha256':_identity(RECIPE),
            'reference_variant':reference_images.get('variant','frozen') if isinstance(reference_images,dict) else 'frozen',
            'candidate_variant':candidate_images.get('variant','unspecified') if isinstance(candidate_images,dict) else 'unspecified',
            'observed_camera_id':cameras['observed_camera_id'],'observed_score_percent':observed,
            'novel_score_percent':sum(novel)/novel_eligible if novel_eligible and len(novel)==novel_eligible else None,'views':views}


def main(argv=None):
    """Render a qualified source from a frozen rig using the installed wheel.

    Example::

        python -m simready_eval.render --source asset.usd --cameras cameras.json \
            --output renders --timeout 180 > render-receipt.json

    Comparison role writes four RGB PNGs and four independent coverage masks.
    Observation role renders the separate fifth camera into its own output
    directory. The all role renders all five cameras in one extraction/job. All five cameras and the scientific recipe remain frozen.
    """
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--package-root',type=Path,help='Caller-qualified closure root; defaults to source parent')
    parser.add_argument('--cameras',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--timeout',type=float,default=180.)
    parser.add_argument('--role',choices=('comparison','observation','all'),default='comparison')
    arguments=parser.parse_args(argv)
    try:
        if not math.isfinite(arguments.timeout) or arguments.timeout<=0:
            raise ValueError('timeout must be positive and finite')
        receipt=render_asset(arguments.source,arguments.cameras,arguments.output,
                             timeout=arguments.timeout,role=arguments.role,package_root=arguments.package_root)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        receipt={'schema':'simready-render-receipt-v1','status':'unknown','images':[],
                 'reason':'invalid_render_input','error_type':type(exc).__name__,'error':str(exc)}
        print(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False))
        return 2
    print(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False))
    expected=5 if arguments.role=='all'else 4 if arguments.role=='comparison'else 1
    return 0 if receipt['status']=='measured' and len(receipt['images'])==expected else 1


if __name__=='__main__':
    raise SystemExit(main())
