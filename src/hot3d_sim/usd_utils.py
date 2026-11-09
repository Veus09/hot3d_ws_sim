"""Tiện ích USD thuần pxr (không cần chạy Isaac Sim)."""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def mesh_to_usd(path, vertices, faces, color=(0.7, 0.7, 0.7), name="obj",
                mass=0.2, friction=0.8, restitution=0.05, collision="convexDecomposition"):
    """Ghi 1 vật rigid: /<name> (Xform, RigidBody, mass) / mesh (collision). Đơn vị mét, Z-up."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
    m = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    f = np.asarray(faces, dtype=np.int32)
    m.CreateFaceVertexCountsAttr([3] * len(f))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc.CreateApproximationAttr(collision)
    mat = UsdShade.Material.Define(stage, f"/{name}/phys_mat")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(friction); pm.CreateDynamicFrictionAttr(friction)
    pm.CreateRestitutionAttr(restitution)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return str(path)
