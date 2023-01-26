from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Charger le modèle
model = load_model('../keras_model.h5')

# Charger les données d'évaluation
eval_data = np.array([image.img_to_array(image.load_img('eval_data/image1.png', target_size=(224, 224)))])
eval_labels = np.array([0]) # 0 pour Yoshi, 1 pour rien

# Évaluation du modèle
eval_loss, eval_acc = model.evaluate(eval_data, eval_labels)
print(f'Evaluation Loss (taux d\'echec) : {eval_loss}, Evaluation Accuracy (taux de réussite): {eval_acc}')
