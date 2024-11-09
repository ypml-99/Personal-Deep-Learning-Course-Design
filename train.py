from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

model = YOLO("yolov10n.yaml")

model.train(
    data="voc_data.yaml",
    epochs=100,                
    imgsz=1024,         
    batch=32,                 
    lr0=0.002,                
    lrf=0.05,                
    weight_decay=0.0005,     
    momentum=0.937,            
    mosaic=0.5,                 
    mixup=0.1,
    multi_scale=True,   
    pretrained=True,           
    amp=True,               
    box=0.05,              
    cls=0.4,               
    dfl=0.6,     
    freeze=10,                
    translate=0.1,        
    scale=0.5,           
    shear=0.2,             
    auto_augment="randaugment" 
)
