# Few-TensoRF demo
This is the replicate demo for Few-TensoRF base on branch "streamlit_nerf" of main source

## Main Source code
- https://github.com/hautran7201/3D-reconstruction.git

## Run streamlit demo
```
streamlit run main.py
``` 

## Run train code manually
```
python -m few_tensorf.train --config few_tensorf/configs/ficus.txt --render_train 1 --export_mesh 1
```

