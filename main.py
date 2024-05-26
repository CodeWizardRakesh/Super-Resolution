import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import gradio as gr
import numpy as np
import os

model_path = "models\\RRDB_ESRGAN_x4.pth"
device = torch.device('cuda')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

def upscale_image(image):
    folder_name = "input_history"
    file_list = []
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_name)
    print(file_list)
                
    #cv2.imwrite("input_history\\comic_input.png", np.array(image))
    image.save("input_history\\" + str(len(file_list)) + ".png")
    device = torch.device('cuda')
    # device = torch.device('cpu')

    input_image_path = "input_history\\" + str(len(file_list)) + ".png"
    output_folder = 'results'

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output_image_path = osp.join(output_folder, osp.basename(input_image_path))
    cv2.imwrite(output_image_path, output)
    return output_image_path

    

demo = gr.Interface(
    upscale_image,
    gr.Image(type="pil"),
    "image",
    flagging_options=["blurry", "incorrect", "other"],
)

# btn = gr.Button(value = "ESRGAN")
# demo.add(btn)
demo.launch()
