# CAD2Tech

## Folder Descriptions
The 'abc_dataset' folder contains data from the ABC dataset (https://archive.nyu.edu/handle/2451/44309);

The 'example_material' folder contains the following:
   - The folders **'collages_3'**, **'collages_4'**, **'collages_6'** contain image collages consisting of 3, 4, or 6 images, respectively, created using Isomap.
   - The **'json_standard'** folder contains reference solutions in the form of JSON files.
   - The **'prompts'** folder contains prompts for generating JSON files describing mechanical processing planning processes for parts.   
   - The **'rendered_imgs'** folder contains multi-view images of the part.
   - The **'example_material/equipment_iso.csv'** file contains data of normative documents developed by the International Organization for Standardization.
   - The **'example_object_path.pkl'** file contains an array of all files in .obj format.

## Framework launch process
First, you need to run the scripts **'render_script_type1.py'** and **'render_script_type2.py'** to render the 3D model into 28 images.
   - Blender (https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip) must be installed and placed in the root of the project beforehand.
   - Commands to run the scripts:
```
# Create 8 views
blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'

# Create 20 views
blender -b -P render_script_type2.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```

Next, you should run the **'framework.py'** file, which performs the following tasks:
   - Selects 3, 4, or 6 different images of the model from various angles and combines them into a single image.
   - Generates JSON files describing the planning processes for mechanical part machining. The inputs for the model are a prompt and the previously created multi-view collage. The Pixtral 12B, Qwen2.5-VL-72B, and Qwen-VL-Max models are used for JSON generation.
   - Evaluates the quality of the generated JSON files against specified criteria using the GPT-4o mini model.
