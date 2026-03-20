import joblib
from clasificador2 import train_and_predict

if __name__ == "__main__":
    try:
        modelo_cargado = joblib.load('modelo_clasificacion.pkl')
        vectorizer_cargado = joblib.load('vectorizer.pkl')
        print("Modelo y vectorizador cargados exitosamente.")
    except FileNotFoundError:
        modelo_cargado, vectorizer_cargado = None, None
        print("No se encontró el modelo preentrenado. El modelo será entrenado desde cero.")

    print("\nBienvenido al clasificador de álbumes. Escribe una letra de una canción y te diré a qué álbum de Taylor Swift pertenece.")
    
    while True:
        lyric = input("\nPor favor, ingresa la letra de la canción (o escribe 'salir' para terminar): ")

        if lyric.lower() == 'salir':
            print("Gracias por usar el clasificador de álbumes. ¡Hasta luego!")
            break

        try:
            album_predicho = train_and_predict(lyric, modelo_cargado, vectorizer_cargado)
            print(f"La letra ingresada pertenece al álbum: {album_predicho}")

            respuesta = input("¿Es correcto? (Sí/No): ")
            if respuesta.lower() == 'sí':
                print("¡Genial! La predicción fue correcta.")
            elif respuesta.lower() == 'no':
                print("Entiendo que la predicción no fue correcta. Si deseas reentrenar el modelo, puedes hacerlo manualmente.")
            else:
                print("Respuesta inválida. Por favor, responde 'Sí' o 'No'.")
        except Exception as e:
            print(f"Ocurrió un error al predecir el álbum: {e}")
