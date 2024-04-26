import sys
import vtk

# Read the .stl file
filename = sys.argv[1]
a = vtk.vtkSTLReader()
a.SetFileName(filename)
a.Update()
a = a.GetOutput()

# Write the .vtk file
filename = filename.replace('.stl', '.vtk')
b = vtk.vtkPolyDataWriter()
b.SetFileName(filename)
b.SetInputData(a)
b.Update()