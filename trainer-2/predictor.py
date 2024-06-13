import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_texts(model_path, texts):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

    predicted_label_indices = predicted_labels.tolist()

    label_names = [
                "Ocio y restauracion",
                "Alimentacion",
                "Compras",
                "Reintegros",
                "Seguros & medicos",
                "Ingresos",
                "Suscripciones",
                "Descartado",
                "Vehiculo y transporte",
                "Alquiler",
                "Company",
                "Limpieza & Babysitter",
                "Pisos / Inversiones",
                "ONGs",
                "Educacion",
                "Luz & Electricidad & Basuras",
                "Tasas, comisiones e impuestos",
        ] 

    predicted_class_names = [label_names[label] for label in predicted_label_indices]
    return predicted_class_names


# Example usage
texts = [
"Pago en LA FLECA BARCELONA ES",
"Pago en VIVARI BARCELONA ES",
"Pago en DECO HOME MERIDIANA BARCELONA ES",
"Reintegro efectivo tarjeta CASHZONE BARCELONA ES",
"Pago en COM.MEN.FRUTAS VERDURAS BARCELONA ES",
"Pago en CONDIS SARDENYA 45 BARCELONA ES",
"Pago en CARROT CAFE S.L. BARCELONA ES",
"Pago en WOMEN SECRET BARCELONA ES",
"Pago en CAPRABO 7771 CRTA ANTIGUABARCELONA ES",
"Pago en CHENPAN BARCELONA ES",
"Traspaso interno recibido",
"Pago en CONDIS SARDENYA 45 BARCELONA ES",
]
predictions = predict_texts("./models_created/finance-trainer_oberta-base-bne", texts)

print(predictions)