import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns
import random
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.cm as cm
from rembg import remove
import re





st.title("PlantScan AI")

data = pd.read_csv('data.csv')

data.loc[data['malade'] == False, 'nom_maladie'] = 'healthy'

st.sidebar.title("Sommaire")

img_final = None

pages = ["Introduction", "Exploration", "Dataset sans background", "Modélisation","Démonstration","Interprétabilité","Conclusion"]

page = st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
    
    st.write("### Introduction")
    
    st.image("plante.png")
        
    st.markdown("L’objectif de ce projet est classifier l’espèce d’une plante dans une image. Une fois la classification faite, vous pourrez savoir si elle est malade ou saine et identifier la maladie le cas échéant.")

    st.markdown("L'application sera donc capable à partir d’une image prise par l'appareil photo de donner une succession d’informations à l’utilisateur.")
                
    st.markdown("Le jeu de données est le [New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset).")
    
    if st.button("Afficher quelques plantes aléatoirement"):
        selected_data = data.groupby('espece').first().reset_index()
        selected_data = selected_data.sample(frac=1).reset_index(drop=True)

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        fig.tight_layout()

        for index, row in selected_data.iterrows():
            img_path = row['img_path']
            espece = row['espece']

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax = axs[index // 3, index % 3]
            ax.imshow(img_rgb)
            ax.set_title(espece)
            ax.axis('off')

            if index == 5:
                break

        st.pyplot(fig)

elif page == pages[1]:
    st.write("### Exploration")
    
    st.write("##### Le jeu de données")
    st.write("Nous avons 14 espèces différentes. Parmi ces espèces, nous avons des plantes malades et des plantes saines. Pour les plantes malades, nous avons plusieurs types de maladie par espèce. Le nombre total de classes est de 38.")
    st.write("L'ensemble de données est constitué de deux parties :")
    st.write(" - Une partie Train (70296 images) contenant 38 dossiers, un dossier pour chaque maladie, avec environ 1900 images par dossier")
    st.write(" - Une partie Valid (17571) contenant 38 dossiers avec environ 470 images par dossier.")

    st.write("##### Nombre d'images par classe")

    root_path = r'C:\Users\aissa\Desktop\Projet\Exploration\new_plant_diseases_dataset\New Plant Diseases Dataset(Augmented)\train'

    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    image_counts = {}
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        image_count = 0
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_count += 1
        subdir_name = os.path.basename(dirpath)
        if subdir_name not in image_counts:
            image_counts[subdir_name] = image_count
        else:
            image_counts[subdir_name] += image_count
    
    fig = plt.figure(figsize=(16, 10))
    plt.bar(range(len(image_counts)), list(image_counts.values()), color=['r', 'g', 'b', 'y', 'c', 'm'])
    
    plt.xticks(range(len(image_counts)), list(image_counts.keys()), rotation=90)
    
    plt.gca().xaxis.set_tick_params(labelsize=14)
    plt.gca().xaxis.set_ticklabels(list(image_counts.keys()), rotation=90)
    
    plt.title('Nombre d\'images par classe', fontsize=18)
    plt.xlabel('Classe' , fontsize=16)
    plt.ylabel('Nombre d\'images', fontsize=16)
    st.pyplot(fig)
    
    st.write("##### DataFrame")
    st.dataframe(data.head())
    
    st.write("##### Distribution des espèces")

    counts = data['espece'].value_counts()

    num_colors = len(counts)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    fig = plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color = colors)
    plt.title("Distribution des espèces")
    plt.xlabel("Espèces")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=45, size = 6)
    st.pyplot(fig)
    
    st.write("Nous voyons que l'espèce Tomato est la plus représentée, en effet c'est l'espèce qui contient le plus de dossiers.")

    
    st.write("##### Plantes malades et saines")

    counts = data['malade'].value_counts()

    num_colors = len(counts)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    fig = plt.figure(figsize=(8, 6))
    counts.plot(kind='bar', color = colors)
    plt.title("Distribution des espèces")
    plt.xlabel("Espèces")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=45, size = 9)
    st.pyplot(fig)
    
    st.write("Nous constatons qu'il y a plus de plantes malades que de plantes saines.")

    st.write("##### Histogramme de la distribution des pixels (moyenne par classe)")
    # Le temps d'exécution pour ce graphique étant trop long pour la soutenance une capture d'écran a été partagée
    st.write("Affichage pour quelques classes.")

    st.image("Histogramme de la distribution des pixels (moyenne par classe).png")

   # st.write("##### Histogramme d'intensité (moyenne par classe par canal R, G et B)")
    # Le temps d'exécution pour ce graphique étant trop long pour la soutenance une capture d'écran a été partagée

   # st.write("Affichage une quelques classe.")

   # st.image("Histogramme d'intensité (moyenne par classe par canal R, G et B).png")


elif page == pages[2]:
    
    st.write("### Dataset sans backround")
    st.write("Nous nous sommes interrogés sur les performances des futurs modèles :")
    st.write("Seront-ils plus performants si nous enlevons les background des images ?")
    st.write("Nous avons donc regénéré le dataset sans le background des images grâce à la librairie rembg :")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image("00img_bg.jpg")

    with col2:
        if st.button("Enlever le background"):
            st.image("00img_no_bg.png")
            
    st.write("Les modélisations ont toutes été effectuées avec et sans background.")




elif page == pages[3]:
    
    st.write("### Modélisation")
    
    st.write("Nous avons effectué trois modélisations :")
    st.markdown("- Identification de l'espèce")
    st.markdown("- Détection de maladie")
    st.markdown("- Identification de la maladie")
    st.write("Pour cela nous avons réaliser des modèles de Machine Learning et de Deep Learning.")

    
    st.write("##### Machine Learning")

    st.write("Après plusieurs essaie (Support Vector Machine (SVM), Regression logistique et Random Forest),")
    st.write("nous avons utilisé des modèles avec undersampling et oversampling comme rééchantillonage.")
    st.write("Nous avons décidé de garder les modèles Random Forest qui donnaient les meilleurs résultats.")
    st.image('rapport_classification.png',caption='Rapport de classification pour un Random Forest sur l\'identification de l\'espèce')
    
    st.write("##### Deep Learning")

    st.write("Nous avons réalisé des modèles CNN qui sont bien adaptés aux images, ainsi que des modèles de Transfer Learning. Après plusieurs essais, VGG16 s'est avéré donner les meilleurs scores.")
    st.image('matrice_confusion.png',caption='Matrice de confusion pour un CNN sur l\'identification de la maladie')
    df__ = pd.DataFrame(data.nom_maladie.unique(), columns=['nom_maladie'])
    df__.index = np.arange(len(df__))
    st.table(df__.T)



elif page == pages[4]:

    st.write("### Démonstration")

    uploaded_image = st.file_uploader("Télécharger une image", type=["png", "jpg", "jpeg"])

    
    img_final = uploaded_image

    
    if uploaded_image is not None:
 
        img = Image.open(uploaded_image)
        img = img.resize((256, 256))  
        img_arr = img
        
        def extract_features(img):
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        def predict(image):
            img_array = img_to_array(image)
            img_array = preprocess_input(img_array[np.newaxis, ...])

            prediction = model.predict(img_array)
            return prediction
        
        col1, col2 = st.columns(2)

        with col1 : 
            st.image(img, channels="RGB")
        
        input_shape = (256, 256, 3)  
        with col2 :
            with_background = st.radio("Arrière-plan de l'image :", ("Avec background", "Sans background"))
        
            if with_background == "Sans background":
                img = img.convert("RGBA")
                img = Image.fromarray(remove(np.array(img)))           
                st.image(img, channels="RGB")
            else : 
                        img = Image.open(uploaded_image)
        nom = uploaded_image.name
        nom = re.sub(r'\..+', '', nom)

        resultat = re.findall(r'([A-Z][a-z]+)', nom)
        espece = resultat[0] if resultat else ''
        maladie = ' '.join(resultat[1:]) if len(resultat) > 1 else ''
        est_malade = maladie != "Healthy"


        data_ = {'Espèce': [espece], 'Est Malade': [est_malade], 'Maladie': [maladie]}
        df_ = pd.DataFrame(data_)

        st.dataframe(df_)

        model_type = st.selectbox("Type de modélisation :", ("Identification de l'espèce", "Malade ou pas", "Nom de la maladie"))

        model_options = {
            "Random Forest": "RF",
            "Convolutional Neural Network (CNN)": "CNN",
            "Transfer Learning (TL) VGG16": "TL"
        }
        selected_model = st.selectbox("Modèle :", list(model_options.keys()))

        if st.button("Lancer la modélisation"):
            if with_background == "Avec background":
                if model_type == "Identification de l'espèce":
                    
                    if selected_model == "Random Forest":
                        rf_id = joblib.load('Modèles\model_rf_Espèces.joblib')

                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)

                        img_features = extract_features(image)
                        rf_id_pred = rf_id.predict([img_features])[0]
                        class_probabilities = rf_id.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_id.classes_ == rf_id_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]
                        st.write("Prédiction de l'espèce (avec background) : ", rf_id_pred)
                        st.write("Probabilités d'appartenance à chaque classe : ", pred_class_probability)

                        
                    elif selected_model == "Convolutional Neural Network (CNN)":
                        num_classes = 14

                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(num_classes, activation='softmax')
                        ])

                        model.load_weights('cnn_id_espece.h5')

                        labels = data['espece'].unique()

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction de l'espèce (avec background) : ", predicted_label)
                        st.write("Probabilité d'appartenance à la classe prédite : ", predicted_probability)

                    elif selected_model == "Transfer Learning (TL) VGG16":
                        
                        model = tf.keras.models.load_model('tl_vgg16_species.h5')
                        pil_image = Image.open(uploaded_image)
                        labels = data['espece'].unique()

                        pil_image = pil_image.resize((256, 256))

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction de l'espèce (avec background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                elif model_type == "Malade ou pas":
                    
                    if selected_model == "Random Forest":
                        rf_malade = joblib.load('rf_model_malade.pkl')
                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)

                        img_features = extract_features(image)
                        
                        rf_malade_pred = rf_malade.predict([img_features])[0]
                        class_probabilities = rf_malade.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_malade.classes_ == rf_malade_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]

                        st.write("Prédiction (malade ou pas) (avec background) : ", rf_malade_pred)
                        st.write("Probabilité d'appartenance à la classe prédite : ", pred_class_probability)
                        
                    elif selected_model == "Convolutional Neural Network (CNN)":
                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])

                        model.load_weights('cnn_id_malade.h5')

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = (predictions > 0.5).astype(int)[0][0]
                        predicted_label = "Malade" if predicted_class == 1 else "Non malade"
                        predicted_probability = predictions[0][0]

                        st.write("Prédiction (malade ou pas) (avec background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)


                        
                    elif selected_model == "Transfer Learning (TL) VGG16":
                        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

                        for layer in base_model.layers:
                            layer.trainable = False

                        x = base_model.output
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(128, activation='relu')(x)

                        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

                        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

                        model.load_weights('tf_VGG16_id_malade.h5')

                        image = load_img(uploaded_image, target_size=(256, 256))
                        img_array = img_to_array(image)
                        img_array = preprocess_input(img_array[np.newaxis, ...])

                        prediction = model.predict(img_array)
                        predicted_label = "Malade" if prediction < 0.5 else "Pas malade"
                        predicted_probability = prediction[0][0]

                        st.write("Prédiction (malade ou pas) (avec background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)


                        
                elif model_type == "Nom de la maladie":
                    
                    if selected_model == "Random Forest":
                        rf_nom_maladie = joblib.load('rf_model_nom_maladie.pkl')
                        
                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)

                        img_features = extract_features(image)
                        rf_nom_maladie_pred = rf_nom_maladie.predict([img_features])[0]   
                        
                        class_probabilities = rf_nom_maladie.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_nom_maladie.classes_ == rf_nom_maladie_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]


                        st.write("Prédiction du nom de la maladie (avec background) : ", rf_nom_maladie_pred)
                        st.write("Probabilité d'appartenance à la classe prédite : ", pred_class_probability)

                    elif selected_model == "Convolutional Neural Network (CNN)":
                        num_classes = 21

                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(num_classes, activation='softmax')
                        ])

                        model.load_weights('cnn_id_nom_maladie.h5')

                        labels = data['nom_maladie'].unique()

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction du nom de la maladie (avec background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                    elif selected_model == "Transfer Learning (TL) VGG16":
                        
                        model = tf.keras.models.load_model('tl_vgg16_disease.h5')
                        pil_image = Image.open(uploaded_image)
                        labels = data['nom_maladie'].unique()

                        pil_image = pil_image.resize((256, 256))

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction du nom de la maladie (avec background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

            else:  # Sans background
                
                if model_type == "Identification de l'espèce":
                    if selected_model == "Random Forest": 
                        
                        rf_id_no_bg = joblib.load('rf_model_id_espece_no_bg.pkl')
                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)
                        img_features = extract_features(image)

                        rf_id_no_bg_pred = rf_id_no_bg.predict([img_features])[0]

                        
                        class_probabilities = rf_id_no_bg.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_id_no_bg.classes_ == rf_id_no_bg_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]


                        
                        
                        
                        st.write("Prédiction de l'espèce (sans background) : ", rf_id_no_bg_pred)
                        st.write("Probabilités d'appartenance à chaque classe : ", pred_class_probability)

                        
                    elif selected_model == "Convolutional Neural Network (CNN)":
                        num_classes = 14

                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(num_classes, activation='softmax')
                        ])

                        model.load_weights('cnn_id_espece_no_bg.h5')

                        labels = data['espece'].unique()

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))
                        pil_image = pil_image.convert("RGB")

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction de l'espèce (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                    elif selected_model == "Transfer Learning (TL) VGG16":
                        
                        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

                        num_classes = 14
                        labels = data['espece'].unique()

                        for layer in base_model.layers:
                            layer.trainable = False

                        x = base_model.output
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(128, activation='relu')(x)
                        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

                        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

                        model.load_weights('tf_VGG16_id_espece_no_bg.h5')

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))
                        pil_image = pil_image.convert("RGB")

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction de l'espèce (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                elif model_type == "Malade ou pas":
                    
                    if selected_model == "Random Forest":
                        
                       
                        
                        rf_maladed_no_bg = joblib.load('rf_model_malade_no_bg.pkl')
                        
                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)

                        img_features = extract_features(image)
                        
                        rf_malade_no_bg_pred = rf_maladed_no_bg.predict([img_features])[0]
                        
                        class_probabilities = rf_maladed_no_bg.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_maladed_no_bg.classes_ == rf_malade_no_bg_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]

                        st.write("Prédiction (malade ou pas) (avec background) : ", rf_malade_no_bg_pred)
                        st.write("Probabilité d'appartenance à la classe prédite : ", pred_class_probability)
                        
                        
                    elif selected_model == "Convolutional Neural Network (CNN)":
                        
                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])

                        model.load_weights('cnn_id_malade_no_bg.h5')

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))
                        pil_image = pil_image.convert("RGB")

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = (predictions > 0.5).astype(int)[0][0]
                        predicted_label = "Malade" if predicted_class == 1 else "Non malade"
                        predicted_probability = predictions[0][0]

                        st.write("Prédiction (malade ou pas) (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                    elif selected_model == "Transfer Learning (TL) VGG16":
                        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
                        input_shape = (256, 256, 3)

                        for layer in base_model.layers:
                            layer.trainable = False

                        x = base_model.output
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(128, activation='relu')(x)
                        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

                        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
                        model.load_weights('tf_VGG16_id_malade_no_bg.h5')

                        image = load_img(uploaded_image, target_size=(256, 256))
                        img_array = img_to_array(image)
                        img_array = preprocess_input(img_array[np.newaxis, ...])

                        prediction = model.predict(img_array)
                        predicted_label = "Malade" if prediction < 0.5 else "Pas malade"
                        predicted_probability = prediction[0][0]

                        st.write("Prédiction (malade ou pas) (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                elif model_type == "Nom de la maladie":
                    
                    if selected_model == "Random Forest":
                        
                        rf_nom_maladie_no_bg = joblib.load('rf_model_nom_maladie_no_bg.pkl')
                        
                        pil_image = Image.open(uploaded_image)

                        image = np.array(pil_image)

                        img_features = extract_features(image)
                        
                        rf_nom_maladie_no_bg_pred = rf_nom_maladie_no_bg.predict([img_features])[0]
                        
                        class_probabilities = rf_nom_maladie_no_bg.predict_proba([img_features])[0]

                        # Index de la classe prédite
                        pred_class_index = np.where(rf_nom_maladie_no_bg.classes_ == rf_nom_maladie_no_bg_pred)[0][0]

                        # Probabilité de la classe prédite
                        pred_class_probability = class_probabilities[pred_class_index]


                        
                        st.write("Prédiction du nom de la maladie (sans background) : ", rf_nom_maladie_no_bg_pred)
                        st.write("Probabilité d'appartenance à la classe prédite : ", pred_class_probability)

                    elif selected_model == "Convolutional Neural Network (CNN)":
                        num_classes = 21

                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                            tf.keras.layers.Dense(num_classes, activation='softmax')
                        ])

                        model.load_weights('cnn_id_nom_maladie_no_bg.h5')

                        labels = data['nom_maladie'].unique()

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))
                        pil_image = pil_image.convert("RGB")

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction du nom de la maladie (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

                        
                    elif selected_model == "Transfer Learning (TL) VGG16":
                        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
                        num_classes = 21
                        labels = data['nom_maladie'].unique()

                        for layer in base_model.layers:
                            layer.trainable = False

                        x = base_model.output
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dense(128, activation='relu')(x)
                        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

                        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
                        model.load_weights('model_tf_VGG16_id_nom_maladie_no_bg.h5')

                        pil_image = Image.open(uploaded_image)
                        pil_image = pil_image.resize((256, 256))
                        pil_image = pil_image.convert("RGB")

                        img_array = image.img_to_array(pil_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array /= 255.0

                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        predicted_label = labels[predicted_class]
                        predicted_probability = predictions[0][predicted_class]

                        st.write("Prédiction du nom de la maladie (sans background) : ", predicted_label)
                        st.write("Probabilité de la classe prédite : ", predicted_probability)

    

       # if st.button("Afficher le tableau des modélisations"):
       #     st.image('tableau.png')

       
        
       
    
    
if page == pages[5] :

    st.write("### Interprétabilité")

    st.write("##### Bilan de la modélisation")
    st.image("tableau.png")
    
    st.write("##### Inrerprétabilité du modèle de Transfer Learning VGG16 : comment apprend-t-il ? ")
    st.write("###### Schéma des couches de VGG16")

    st.image("VGG16.png" , caption = 'VGG16')
    
    st.write("###### Schéma explicatif de Grad-Cam")

    st.image("GRADCAM.png", caption = 'Grad-Cam')


    st.write("Les régions rouges dans les images Grad-Cam permettent de mettre en évidence les régions discriminatoires les plus importantes pour la prédiction du modèle.")

   
    col02, col01 = st.columns(2)
    with col01:
        images = [
            Image.open(r'grad_cam\1.png'),
            Image.open(r'grad_cam\2.png'),
            Image.open(r'grad_cam\3.png'),
            Image.open(r'grad_cam\4.png'),
            Image.open(r'grad_cam\5.png'),
            Image.open(r'grad_cam\6.png'),
            Image.open(r'grad_cam\7.png'),
            Image.open(r'grad_cam\8.png'),
            Image.open(r'grad_cam\9.png'),
            Image.open(r'grad_cam\10.png'),
            Image.open(r'grad_cam\11.png'),
            Image.open(r'grad_cam\12.png'),
            Image.open(r'grad_cam\13.png'),
            Image.open(r'grad_cam\14.png'),
            Image.open(r'grad_cam\15.png'),
            Image.open(r'grad_cam\16.png'),
            Image.open(r'grad_cam\17.png'),
            Image.open(r'grad_cam\18.png'),
            Image.open(r'grad_cam\19.png')
        ]

        rows = 5
        cols = 4

        selected_image_index = st.slider('Couches du modèle avec background', 0, len(images), 1)

        selected_row = (selected_image_index - 1) // cols
        selected_col = (selected_image_index - 1) % cols

        index = 0
        for i in range(rows):
            cols_list = st.columns(cols)
            for j in range(cols):
                if index < len(images):
                    image = images[index]
                    if i < selected_row or (i == selected_row and j <= selected_col):
                        cols_list[j].image(image, use_column_width=True)
                    index += 1

    with col02:
        images = [
            Image.open(r'grad_cam_no_bg\1.png'),
            Image.open(r'grad_cam_no_bg\2.png'),
            Image.open(r'grad_cam_no_bg\3.png'),
            Image.open(r'grad_cam_no_bg\4.png'),
            Image.open(r'grad_cam_no_bg\5.png'),
            Image.open(r'grad_cam_no_bg\6.png'),
            Image.open(r'grad_cam_no_bg\7.png'),
            Image.open(r'grad_cam_no_bg\8.png'),
            Image.open(r'grad_cam_no_bg\9.png'),
            Image.open(r'grad_cam_no_bg\10.png'),
            Image.open(r'grad_cam_no_bg\11.png'),
            Image.open(r'grad_cam_no_bg\12.png'),
            Image.open(r'grad_cam_no_bg\13.png'),
            Image.open(r'grad_cam_no_bg\14.png'),
            Image.open(r'grad_cam_no_bg\15.png'),
            Image.open(r'grad_cam_no_bg\16.png'),
            Image.open(r'grad_cam_no_bg\17.png'),
            Image.open(r'grad_cam_no_bg\18.png'),
            Image.open(r'grad_cam_no_bg\19.png')
        ]

        rows = 5
        cols = 4

        selected_image_index = st.slider('Couches du modèle sans background', 0, len(images), 1)

        selected_row = (selected_image_index - 1) // cols
        selected_col = (selected_image_index - 1) % cols

        index = 0
        for i in range(rows):
            cols_list = st.columns(cols)
            for j in range(cols):
                if index < len(images):
                    image = images[index]
                    if i < selected_row or (i == selected_row and j <= selected_col):
                        cols_list[j].image(image, use_column_width=True)
                    index += 1
    image = Image.open(r'grad_cam\0.png')
    image = image.resize((256, 256))
    image = image.convert('RGB')

    # Chargement du modèle
    input_shape = (256, 256, 3)
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


    grad_model = tf.keras.models.Model([model.input], [model.get_layer('block5_pool').output, model.output])

    # Prétraitement de l'image
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Curseur 
    #weight = st.slider("##### Explication du Grad-CAM", 0.0, 1.0, 0.5, 0.05)

    #with tf.GradientTape() as tape:
    #    conv_outputs, predictions = grad_model(img)
    #    label_idx = tf.argmax(tf.squeeze(predictions)).numpy()  
    #    loss = tf.gather(tf.squeeze(predictions), [label_idx])

   # output = conv_outputs[0]
    #grads = tape.gradient(loss, conv_outputs)[0]
    #gate_f = tf.cast(output > 0, 'float32')
   # gate_r = tf.cast(grads > 0, 'float32')
   # guided_grads = gate_f * gate_r * grads
   # weights = tf.reduce_mean(guided_grads, axis=(0, 1))
   # cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
   # cam = np.maximum(cam, 0)
   # cam = cv2.resize(cam, (64, 64))  
   # heatmap = cam / np.max(cam)

   # colors = cm.jet(heatmap)
   # colors[..., 0] *= weight  # Rouge
   # colors[..., 1] *= 1 - weight  # Vert
   # colors[..., 2] *= 1 - weight  # Bleu

   # overlay = colors[:, :, :3]

   # overlay = cv2.resize(overlay, (img.shape[2], img.shape[1]))

  #  gradcam = (overlay * 255).astype(np.uint8) + cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

  #  col1, col2 = st.columns(2)
  #  with col1:
  #      st.image(image, caption="Image d'origine")
  #  with col2:
  #      st.image(gradcam, caption="Grad-CAM")
        
    

        
elif page == pages[6]:
    
    st.write("### Conclusion")
    
    st.write("Nous avons réaliser des modèles de Machine Learning et Deep Learning pour effectuer trois classifications.")
    st.write("Ces classifications permettent à l'utlisateur à partir d'une image de :")
    st.markdown("- Identifier l'espèce de la plante")
    st.markdown("- Détecter la présence éventuelle d'une maladie")
    st.markdown("- Si la plante est malade, d'identifier la maladie")
    st.image('schema.png', caption = 'Schéma récapitulatif')
             
    #st.write("Également, le modèle peut être utilisé pour détecter et classifier les maladies des plantes,permettant ainsi un diagnostic rapide et précis des problèmes de santé des cultures. Celapeut aider les agriculteurs à prendre des mesures appropriées pour contrôler et prévenir les maladies, améliorant ainsi la santé et le rendement des cultures.")

    st.write("##### Pistes d'amélioration")
    st.markdown("- Collecter des données supplémentaires et augmenter le nombre de classe")
    st.markdown("- Utiliser d'autres modèles")
    
    st.write("##### Utilisation")
    st.write("Cette application peut s'inscrire dans plusieurs processus métiers liés à l'agriculture, à la botanique et à la gestion des cultures.")