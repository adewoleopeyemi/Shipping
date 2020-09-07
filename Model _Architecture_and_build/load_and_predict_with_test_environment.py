from tensorflow.keras.models import load_model

small_model = load_model("trainedmodelsh5/deploytestnudity.h5")
deployment_model = load_model("trainedmodelsh5/nuditydetectionalgorithm.h5")

def test():
    ' write your tests here'