# Keypoint annotation tool 

This tool was developed for the purpose of creating new keypoint regressors, 
that directly integrate into the HMR framework. 

### Requirements
- Python > 3.6

#### setup virtual environment
If you did not setup virtualenvwrapper checkout the description in [README](../README.md) 
```
mkvirtualenv keypoint_tool
workon keypoint_tool
pip install -U pip
```

### install requirements 
```
pip install -r keypoint_marker/requirements.txt
```

### run
```
python main.py
```

## Annotation Process
1. select a mesh from the dropdown
2. specify which model should be used via toggles (neutral, female, male)
3. specify a new keypoint by selecting a set of triangles  
    - just click on the surface 
    - a keypoint is shown when at least 10 triangles are selected 
    - press `w` to visualize keypoint 
    - min. triangles can be changed in Settings
4. repeat step `3` for at least 5 meshes
    - min. meshes can be changed in Settings
5. fill in a name for the regressor
6. hit the `Convert` button 

![tool](../images/tool.png)


## Commands

|command||
|---|---|
|mouse drag |rotates the view|
|ctrl/cmd + mouse drag |pans the view |
|mouse wheel |zooms the view |
|z |returns to the base view |
|v |toggles smoothing |
|j |toggles joints |
|r |toggles camera rays |
|w |toggles wireframe mode |
|c |toggles backface culling |
|f |toggles between fullscreen and windowed mode |
|m |maximizes the window |
|q |quit the tool |

## Known Issues

- Toggle smoothing can cause change of the mesh topology.
This can lead to failing regressor generation!
