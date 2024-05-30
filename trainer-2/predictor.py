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
"Pago en PAYPAL *friedemann.str 35314369001 ES",
"Pago en MARMARIS Restaurant AMSTERDAM ZUINL",
"Pago en ALBERT HEIJN 2236",
"Pago en TAXI JAPPIE",
"Recibo ENDESA X SERVICIOS S.L.",
"Transferencia internacional emitida Taschengeld",
"Pago en AREAS LOIU ES",
"Pago en BILBBO INTERMODAL BILBAO ES",
]
predictions = predict_texts("./models_created/finance_trainer", texts)

print(predictions)