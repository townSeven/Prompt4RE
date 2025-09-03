from PIL import Image
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from parse_args import args

# ==================== 工具函数 ====================
def load_and_resize(image_path, size=(640, 640)):
    """加载并缩放图片"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img

# ==================== 模型加载 ====================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data4/jiangchangyang/Qwen-VL-master/Qwen2.5_VL",
    torch_dtype=torch.float16,
    device_map="auto"   # 自动分配多卡
)
processor = AutoProcessor.from_pretrained("/data4/jiangchangyang/Qwen-VL-master/Qwen2.5_VL")

# ==================== 读取 JSON ====================
json_file = "/data5/luyisha/guozitao/LLM_region/data/prompt_data.json"
with open(json_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

embeddings = []
task = args.task
if task == "crime":
    task_prompt = "You are a top expert in urban criminology and urban planning. Your task is to analyze various data of a city area and deeply understand the potential characteristics related to crime risk in that area.\n[Input Information]\nGiven are three types of information:\n1. Satellite images: Please focus on the macroscopic layout of the area, such as: land use types (residential areas, commercial areas, industrial areas, open spaces), building density, road network structure (grid-like or disordered), connectivity with surrounding areas (such as central commercial areas, transportation hubs), green coverage rate, etc.\n2. Street view images: Please focus on the micro environment of the area, such as: the condition of buildings (such as graffiti on walls, broken windows), the condition of public facilities (such as whether the streetlights are intact, whether there are surveillance cameras), the cleanliness of the streets, accessibility of sidewalks, signs of commercial activities (such as busy shops or closed and bankrupt ones), etc.\n3. Text information: Contains the latitude and longitude of the area, the address, and POI category information.\n[Your Core Task]\nPlease conduct a step-by-step reasoning to analyze which features and patterns in the multi-modal information are highly correlated with potential urban crime risk. Combine all the information to conduct a comprehensive 'region diagnosis'.\n[Please think and output in the following structure]\n1. Physical environment analysis： Based on satellite and street view images， list all the physical features that may indicate an increase or decrease in crime risk (such as： blind spots without natural surveillance， abandoned buildings， poorly maintained public spaces).\n2. Social and economic background analysis： Based on text data and the inferred information in the images (such as the level of regional prosperity)， analyze the possible social and economic factors that may affect the crime rate (such as： signs of economic recession， population mobility， community cohesion).\n3. Risk comprehensive assessment： Combine all these points to make a preliminary judgment of the overall crime risk level (high， medium， low)， and explain your core reasons. Focus on how the different modalities of information mutually confirm or provide new perspectives.\n4. Conclusion and feature extraction： Finally， please summarize a set of key feature labels that best represent the crime prediction risk of the area (such as： #High-density old residences #Insufficient commercial vitality #Insufficient lighting #Low-income population concentration #Near transportation hubs).\nPlease focus your thinking on the structured analysis above. Your final output should be this detailed analysis report."
elif task == "checkIn":
    task_prompt = "You are a top urban planner and expert in social media behavior analysis. Your task is to deeply analyze the multimodal data of a city area to assess its attractiveness to people and predict the check-in behaviors it may trigger.\n[Input Information]\nGiven are three types of information:\n1. Satellite images: Please analyze the functional attributes and accessibility of the area from a macro perspective. For example: Is it a residential area, commercial area, industrial area, or recreational area? Are there any obvious attractions (such as parks, squares, landmark buildings)? Is the road network dense? Is there convenient connection to major roads, highways, or transportation hubs (such as subway stations, train stations)?\n2. Street scene images: Please analyze the visual appeal and social atmosphere of the area from a micro perspective. For example: Is the exterior of the buildings aesthetically pleasing or distinctive? Are there popular stores, cafes, restaurants (pay attention to signs and windows)? Is the sidewalk spacious and clean? Are public spaces (such as squares, benches) well-designed? Can you see a dense flow of people, outdoor seating, or entertainment facilities? Is the overall environment safe, comfortable, and worth taking pictures and sharing?\n3. Text information: Contains the latitude and longitude of the area, the address, and POI category information.\n[Your Core Task]\nPlease conduct a step-by-step reasoning, comprehensively analyze the above information, and identify all the driving factors that may prompt people to visit the area and trigger check-in behaviors. Your analysis should focus on the 'attractiveness' of the area.\n[Please think and output in the following structure]\n1. Destination Attractiveness Analysis:\n● Based on all the information, list the specific locations or features within the area that are most likely to be check-in destinations (for example: unique buildings, historical landmarks, popular restaurants, large shopping centers, beautiful coastal parks).\n● Determine the function of the area (what is it for? Shopping, work, dining, or entertainment?) \n2. Accessibility and Convenience Analysis:\n● Analyze how people can reach here (Is the transportation convenient? Are there parking lots, subway stations, or bus stops?) \n● Analyze the pedestrian-friendliness within the area (whether it is easy to walk and explore?) \n3. Analysis of Environment and Social Atmosphere:\n● Assess the appearance quality and comfort level of this area (is it a pleasant place that people would like to stay and take photos to share?) \n● Assess its social vitality (Is it lively? Does it look suitable for socializing? Does it have the potential to host events?) \n4. Comprehensive Evaluation and Feature Extraction:\n● Ultimately, please summarize the overall attractiveness level of this area (high, medium, or low) and explain the key reasons.\n● Most importantly, output a set of key feature labels that best represent the potential for check-in predictions in this area (such as: #commercial center #cuisine cluster #subway above-ground #pedestrian-friendly #hip landmark #park green space #high-density office area #active night economy).\nPlease focus your thinking on the above-structured analysis. Your final output should be this detailed analysis report."
elif task == "serviceCall":
    task_prompt = "You are an experienced urban infrastructure engineer and public affairs management expert. Your task is to conduct an in-depth analysis of the multimodal data of a city area to assess the status of the public infrastructure and predict the potential demand for public services (Service Calls).\n[Input Information]\nGiven are three types of information:\n1. Satellite images: Analyze the overall layout and environmental exposure of the area from a macro perspective. For example: the estimated age of the area, the regularity of the building layout, the density and distribution of green spaces and vegetation, the complexity of the road network, whether there are large areas of exposed ground or waterlogged areas.\n2. Street scene images: Diagnose the physical condition and maintenance level of the infrastructure from a micro perspective. For example: Are there potholes or cracks in the road surface? Are the paving stones of the sidewalks damaged? Are public facilities such as streetlights, traffic signs, and fire hydrants showing obvious rust or damage? Is the green belt trimmed? Are garbage bins overflowing or are there scattered garbage around? Are the drainage channels unobstructed? Are the construction site fences in compliance?\n3. Text information: Contains the latitude and longitude of the area, the address, and POI category information.\n[Your Core Task]\nPlease conduct a step-by-step reasoning, comprehensively analyze the above information, and identify all potential factors that may indicate a high demand for public services and require municipal intervention. Your analysis should focus on the 'infrastructure health' and 'public service demand pressure' of the area.\n[Please think and output in the following structure]\n1. Physical Infrastructure Condition Diagnosis:\n● Based on street scene and satellite images, list in detail all observed or inferred signs of aging, damage, or inadequate maintenance of the infrastructure (for example: #road surface damage #excessive vegetation growth #rusty public facilities #garbage disposal problems).\n● Assess the prevalence and severity of these issues.\n2. Social Environment and Demand Pressure Analysis:\n● Combine text data and image information (such as building types, vehicle conditions), analyze the social and economic characteristics of the area (for example: Is it a resource-poor community? Does the population structure have a higher demand for specific services such as elderly care?) 。\n● Analyze the wear and tear pressure on infrastructure due to population density and usage intensity (for example: the sewer system in high-density residential areas experiences greater pressure).\n3. Comprehensive assessment of vulnerability and risk:\n● Based on all the above points, determine which types of public service requests (such as road repairs, power failures, fallen trees, pipe blockages) this area has higher vulnerability to.\n● Make a preliminary judgment on the overall risk level of public service requests in this area (high, medium, low), and explain your core reasons. For example: 'This area has dense vegetation and high population density. The risk of tree pruning and power restoration after summer storms is higher.'\n4. Conclusion and feature extraction:\n● Finally, please summarize a set of key feature labels that best represent the predicted risk of public service requests in this area (for example: #Old infrastructure #High population density #Poor road conditions #Insufficient maintenance funds #Forested area #Low-income community #Weakened drainage system).\nPlease focus your thinking on the above structured analysis. Your final output should be this detailed analysis report."

# ==================== 遍历每个 region ====================
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(dataset, desc="Processing regions", unit="region")):
        try:
            # 一个 region = 可能有多张图 + 文字
            messages = [sample]

            # 转换为模型输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            # ===== 对图片缩放 =====
            if image_inputs is not None:
                resized_images = []
                for img in image_inputs:
                    if isinstance(img, str):
                        img = load_and_resize(img)
                    elif isinstance(img, Image.Image):
                        img = img.resize((640, 640), Image.LANCZOS)
                    resized_images.append(img)
                image_inputs = resized_images

            # 构建输入
            inputs = processor(
                text=[text + task_prompt],
                images=image_inputs,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # 前向推理
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                images=inputs.get("pixel_values", None),
                output_hidden_states=True,
                return_dict=True
            )

            # 取最后一层 hidden states
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            sentence_embedding = last_hidden.mean(dim=1)  # [1, hidden_dim]

            embeddings.append(sentence_embedding.cpu().numpy())

            # 主动释放缓存，避免累计显存
            del inputs, outputs, last_hidden, sentence_embedding
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n Error in region {idx}: {e}")
            hidden_dim = model.config.hidden_size
            embeddings.append(np.zeros((1, hidden_dim), dtype=np.float32))
            torch.cuda.empty_cache()

# ==================== 保存为 npy ====================
embeddings = np.vstack(embeddings)  # [N, hidden_dim]
print("Final embeddings shape:", embeddings.shape)

np.save("prompt_embs.npy", embeddings)
print("Saved embeddings to prompt_embs.npy")
