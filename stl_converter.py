import sys
import vtk

import gmsh
import aspose.threed as a3d

filename = "rh_simplified"

gmsh.initialize()
stl = gmsh.merge('assets/'+filename+'.stl')
gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
gmsh.model.mesh.createGeometry()

s = gmsh.model.getEntities(2)
surf = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
vol = gmsh.model.geo.addVolume([surf])
gmsh.model.geo.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.option.setNumber('Mesh.Algorithm', 1)
gmsh.option.setNumber('Mesh.MeshSizeMax', 5)
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)

gmsh.model.addPhysicalGroup(3, [1], 1)
gmsh.model.setPhysicalName(3, 1, "The volume")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("assets/"+filename+".vtk")
gmsh.finalize()

scene = a3d.Scene.from_file("assets/"+filename+".stl")
scene.save("assets/"+filename+".obj")