"""Neutral primitive and numerical regression contract, independent of benchmark harness."""
import math
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from simready_eval.usd import observe, triangulate_polygon
from simready_eval.correspondence import nearest, sample, surface, settings


def primitive(path, name, **values):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    body = UsdGeom.Xform.Define(stage, '/Body')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    item = getattr(UsdGeom, name).Define(stage, '/Body/Surface')
    for key, value in values.items():
        item.GetPrim().GetAttribute(key).Set(value)
    stage.GetRootLayer().Save()
    return observe(path)


@pytest.mark.parametrize('name', ['Cylinder', 'Sphere', 'Cone', 'Capsule', 'Cylinder_1', 'Capsule_1', 'Plane'])
def test_primitive_surface_and_reproducible_selfmatch(tmp_path, name):
    observation = primitive(tmp_path / (name + '.usda'), name)
    tri = observation['triangles']
    assert len(tri) > 0
    assert set(observation['triangle_bodies']) == {'/Body'}
    assert not [x for x in observation['eligibility']['unsupported'] if x.get('geometry')]
    a, index, bary = sample(tri, 128)
    b, index2, bary2 = sample(tri, 128)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(index, index2)
    np.testing.assert_array_equal(bary, bary2)
    result, _ = surface(tri, tri, settings({'samples': 128}), .001)
    assert result['score_percent'] == 100


def test_cylinder_area_volume_and_cap_sampling(tmp_path):
    tri = primitive(tmp_path / 'cylinder.usda', 'Cylinder', radius=2., height=3.)['triangles']
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]), axis=1)/2
    volume = abs(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum())/6
    assert area.sum() == pytest.approx(20*math.pi, rel=.001)
    assert volume == pytest.approx(12*math.pi, rel=.001)
    caps = np.ptp(tri[:,:,2], axis=1) == 0
    assert area[caps].sum()/area.sum() == pytest.approx(.4, abs=.0002)
    _, indices, _ = sample(tri, 8192)
    # A fixed 3-percentage-point allowance is over five binomial standard errors.
    assert abs(np.mean(caps[indices])-.4) < .03


def test_collapsed_distance_and_thin_barycentrics():
    tri = np.array([[[0.,0,0],[1,0,0],[2,0,0]], [[3,0,0],[3,0,0],[3,0,0]]])
    distances, _, bary = nearest([[.5,1,0],[3,0,1]],tri)
    np.testing.assert_allclose(distances,[1,1])
    assert np.isfinite(bary).all()
    thin = np.array([[[0.,0,0],[1,0,0],[.5,1e-12,0]]])
    distance, _, bary = nearest([[.5,5e-13,1e-8]],thin)
    np.testing.assert_allclose(distance,[1e-8],atol=1e-16)
    np.testing.assert_allclose(bary,[[.25,.25,.5]],atol=1e-12)
    with pytest.raises(ValueError):
        nearest([[math.nan,0,0]],thin)


def test_near_coincident_quad_keeps_original_corners():
    d = np.finfo(float).eps * 8
    polygon = np.array([[0.,0,0],[d,0,0],[1,1,0],[0,1,0]])
    triangles = triangulate_polygon(polygon)
    assert set(sum((list(x) for x in triangles),[])) == {0,1,2,3}
    assert len(triangles) == 2


@pytest.mark.parametrize('height', [0., 1.])
def test_capsule1_unequal_radius_is_explicitly_unqualified(tmp_path, height):
    observation = primitive(tmp_path / 'capsule.usda', 'Capsule_1', height=height, radiusBottom=.5, radiusTop=.25)
    assert not len(observation['triangles'])
    assert any('qualification' in x['reason'] for x in observation['eligibility']['unsupported'])


@pytest.mark.parametrize('name', ['Cylinder', 'Sphere', 'Cone', 'Capsule', 'Cylinder_1', 'Capsule_1', 'Plane'])
def test_primitive_composed_transform_units_up_axis_and_material(tmp_path, name):
    from pxr import Sdf, UsdShade
    reference = primitive(tmp_path / 'base.usda', name)
    path = tmp_path / 'composed.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    UsdGeom.SetStageUpAxis(stage, 'Y')
    body = UsdGeom.Xform.Define(stage, '/Owner')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddTranslateOp().Set(Gf.Vec3d(10,20,30))
    child = UsdGeom.Xform.Define(stage, '/Owner/Child')
    child.AddScaleOp().Set(Gf.Vec3f(2,3,4))
    item = getattr(UsdGeom, name).Define(stage, '/Owner/Child/Surface')
    if name != 'Sphere':
        item.GetPrim().GetAttribute('axis').Set('X')
    material = UsdShade.Material.Define(stage, '/Material')
    shader = UsdShade.Shader.Define(stage, '/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    actual = observe(path)
    local = reference['triangles']
    if name != 'Sphere':
        local = local[:,:,[2,0,1]]
    expected = ((local*np.array([2,3,4]) + [10,20,30])*.01)[:,:,[0,2,1]] * [1,-1,1]
    np.testing.assert_allclose(actual['triangles'],expected,atol=1e-14)
    assert set(actual['triangle_bodies']) == {'/Owner'}
    assert actual['materials'][0]['visual:opacity'] == pytest.approx(.35)
    np.testing.assert_array_equal(actual['triangles'],observe(path)['triangles'])


def test_near_pole_mesh_corner_provenance_and_no_snapping(tmp_path):
    path=tmp_path/'pole.usda'
    stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,'Z')
    mesh=UsdGeom.Mesh.Define(stage,'/Mesh')
    d=np.finfo(float).eps*8
    points=np.array([[0.,0,0],[d,0,0],[1,1,0],[0,1,0]])
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateSubdivisionSchemeAttr('none')
    stage.GetRootLayer().Save()
    observation=observe(path)
    assert len(observation['triangles'])==2
    metadata=observation['materials'].groups[0][4]
    assert set(metadata[:,1:].ravel()) == {0,1,2,3}
    for triangle,row in zip(observation['triangles'],metadata):
        np.testing.assert_array_equal(triangle,points[row[1:]])


def test_near_pole_crossing_rejected_without_welding():
    d=np.finfo(float).eps*8
    polygon=np.array([[0.,0,0],[d,d,0],[-1,1,0],[1,1,0]])
    with pytest.raises(ValueError, match='self-intersecting'):
        triangulate_polygon(polygon)


def test_surface_sampling_gives_collapsed_faces_zero_weight():
    triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    collapsed=np.zeros((3,3))
    _,indices,_=sample(np.array([collapsed,triangle]),1024)
    assert np.all(indices==1)
    with pytest.raises(ValueError):
        sample(np.array([triangle])*math.nan,128)
