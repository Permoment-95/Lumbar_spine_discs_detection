DEVICE = 'cpu'
import torch
import os
import cv2
from src.architecture import LumbarHmModel
import streamlit as st
import numpy as np
from src.postprocessing import topk, get_coordinates_hm, annotation_format, intersection_over_union
import matplotlib.pyplot as plt


st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.cpu().numpy(),
                      torch.Tensor: lambda t: t.cpu().numpy(),
                     }, suppress_st_warning=True, allow_output_mutation=True)

def cached_model(encoder_name,decoder_name, encoder_weight, path_to_dump):
    model = LumbarHmModel(encoder = 'timm-efficientnet-b4', decoder = 'MAnet')
    
    model.backbone.load_state_dict(torch.load(path_to_dump, map_location='cpu')['model_state_dict'])
    model.eval();
    model.to(DEVICE)
    return model

model = cached_model('timm-efficientnet-b4','MAnet', None, 'best_model/best.pth')
st.title("Detect vertebral discs on MRI")

uploaded_file = st.file_uploader("Choose an image...")

def get_annotated_image(image, model, source_image, DEVICE):
    out = model(torch.tensor([image.reshape(1,512,512)]).to(DEVICE)).detach().cpu()
    source_shape = source_image.shape
    
    out_resize = torch.tensor(cv2.resize(out.permute(0,2,3,1)[0].numpy(), (source_shape[1], source_shape[0])).reshape(1,source_shape[0],source_shape[1],5)).permute(0,3,1,2)
    
    bboxes, cts = get_coordinates_hm(out_resize)
    _, bboxes = annotation_format(bboxes)
    
    fig, ax = plt.subplots()
    fig.set_figheight(30)
    fig.set_figwidth(30)
    
    ax.imshow(source_image, cmap='gray')
    ax.scatter(bboxes[:,0,0], bboxes[:,0,1], s=100)
    ax.scatter(bboxes[:,1,0], bboxes[:,1,1], s=100)
    ax.scatter(bboxes[:,2,0], bboxes[:,2,1], s=100)
    ax.scatter(bboxes[:,3,0], bboxes[:,3,1], s=100)
    
    fig.canvas.draw()
    annotated_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_img = annotated_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    st.image(annotated_img, caption="After", use_column_width=True)
    

if uploaded_file is not None:
    img = np.load(uploaded_file).astype('float32')
    image_resize = cv2.resize(img, (512, 512))
    image_resize = image_resize / image_resize.max()
    
    st.write("Processing image...")
    
    #st.image(img, caption="After", use_column_width=True)
    image_resize = image_resize / image_resize.max()
    
    get_annotated_image(image_resize, model, img, DEVICE)
    ##except InvalidDicomError:
    ##    img = np.array(Image.open(uploaded_file).convert('L'))
    ##    image_class = copy.copy(img)/255.
    #    
    #st.write("Processing image...")
    #
    #image_resize = cv2.resize(image_resize, (512, 512))
    #
    #get_annotated_image(image_resize,model, img, DEVICE)