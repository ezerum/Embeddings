import cv2
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pymilvus import connections, Collection
import os
import json
from omegaconf import ListConfig
import torch.serialization

# Añadir ListConfig a la lista segura
torch.serialization.add_safe_globals([ListConfig])

# Conectar a Milvus
def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def insert_data(collection, entities):
    insert_result = collection.insert(entities)
    collection.flush()
    print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
    return insert_result

def create_index(collection, field_name, index_type, metric_type, params):
    index = {"index_type": index_type, "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index '{index_type}' creado para el campo '{field_name}'.")

def search_and_query(collection, search_vectors, search_field, search_params):
    collection.load()
    result = collection.search(search_vectors, search_field, search_params, limit=3, output_fields=["source"])
    return result

# Cargar modelos
try:
    # Cargar YOLOv8 con weights_only=True si solo se necesitan los pesos
    yolo_model = YOLO("Modelo YOLOv8 a usar")
except Exception as e:
    print(f"Error cargando YOLOv8: {e}")

# Cargar tu modelo entrenado de Retail Object Recognition con weights_only=True si solo necesitas los pesos
def load_retail_model(model_path):
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=True)  # Cargar solo los pesos
        model.eval()
        return model
    except Exception as e:
        print(f"Error cargando el modelo de Retail Object Recognition: {e}")
        return None

# Definir modelo de Retail Object Recognition
retail_model = load_retail_model('/home/devend/Desktop/Enaide/Milvus_vector_embeddings_1/retail_object_recognition_model.pth')

# Función de preprocesamiento de imagen para Retail Object Recognition
def preprocess_image(image: Image.Image) -> Tensor:
    """Preprocesa una imagen PIL para que sea compatible con el modelo entrenado"""
    image = image.resize((224, 224))  # Tamaño de la imagen esperada por el modelo
    image = np.array(image).astype(np.float32) / 255.0  # Normalizar los valores de la imagen
    image = np.transpose(image, (2, 0, 1))  # Cambiar la dimensión a (canales, altura, ancho)
    image_tensor = torch.tensor(image).unsqueeze(0)  # Convertir a tensor y añadir una dimensión batch
    return image_tensor

# Generar embeddings usando el modelo de Retail Object Recognition
def generate_image_embeddings(images, model):
    """Genera embeddings para un batch de imágenes de PIL."""
    image_tensors = [preprocess_image(image) for image in images]
    image_batch = torch.cat(image_tensors, dim=0)
    
    with torch.no_grad():
        outputs = model(image_batch)  # Pasar el batch de imágenes al modelo
    
    # Realizar pooling sobre las dimensiones espaciales
    embeddings = outputs.mean(dim=[2, 3])
    
    # Normalizar los embeddings 
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()

# Guardar embeddings en archivo JSON
def save_embeddings_to_json(embeddings, output_path):
    embeddings_list = embeddings.tolist()  # Convertir los embeddings a una lista
    with open(output_path, 'w') as f:
        json.dump(embeddings_list, f)
    print(f"Embeddings guardados en {output_path}")

# Procesar video
def process_video(video_path, output_path, collection, json_output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    all_embeddings = []  # Lista para almacenar los embeddings

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objetos usando YOLOv8
        results = yolo_model(frame)
        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = frame[y1:y2, x1:x2]
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                # Generar embeddings
                embeddings = generate_image_embeddings([pil_image], retail_model)
                all_embeddings.append(embeddings)  # Agregar los embeddings a la lista
                
                # Buscar en Milvus
                search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
                search_results = search_and_query(collection, embeddings, "embeddings", search_params)
                
                # Dibujar bounding box 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Mostrar el ID con la distancia más pequeña
                if search_results:
                    min_distance_hit = None
                    for search_result in search_results:
                        for hit in search_result:
                            if min_distance_hit is None or hit.distance < min_distance_hit.distance:
                                min_distance_hit = hit
                    
                    if min_distance_hit:
                        # Obtener el valor de "Source_file" y extraer solo la parte del SKU
                        source_file = min_distance_hit.entity.get('source')
                        sku = source_file.split('/')[1]  # Extraer el SKU después de "Dataset_1/"
    
                        result_text = f"ID similar: {min_distance_hit.id}, Distancia: {min_distance_hit.distance:.2f}, SKU: {sku}" 
                        print(f"ID similar: {min_distance_hit.id}, Distancia: {min_distance_hit.distance:.2f}, SKU: {sku}")
                        cv2.putText(frame, result_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(frame)

    cap.release()
    out.release()

    # Guardar todos los embeddings en un archivo JSON
    save_embeddings_to_json(np.concatenate(all_embeddings), json_output_path)

# Main
connect_to_milvus()

# Cargar la colección existente en lugar de crear una nueva
collection = Collection("GO2_Products")

# Procesar video
video_path = 'path al video que quieras inferir'
output_path = 'path de output'
json_output_path = 'output/embeddings.json'
process_video(video_path, output_path, collection, json_output_path)
