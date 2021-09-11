
# run it with:
# python3 app.py

#import the necessary libraries
from flask import Flask, render_template , request
import pickle

import numpy as np
import pandas as pd
#from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

app = Flask(__name__)

# load the model from disk
model = pickle.load(open('finalized_model_96_new.pkl', 'rb'))

def getAromaticProportion(m):
    aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aromatic = 0
    for i in aromatic_list:
        if i:
            aromatic += 1
    heavy_atom = Lipinski.HeavyAtomCount(m)
    return aromatic / heavy_atom


def predictSingle(smiles, model):
    """
    This function predicts the four molecular descriptors: the octanol/water partition coefficient (LogP),
    the molecular weight (Mw), the number of rotatable bonds (NRb), and the aromatic proportion (AP) 
    for a single molecule
    
    The input arguments are SMILES molecular structure and the trained model, respectively.
    """
    
    # define the rdkit moleculer object
    mol = Chem.MolFromSmiles(smiles)
    
    # calculate the log octanol/water partition descriptor
    single_MolLogP = Descriptors.MolLogP(mol)
    
    # calculate the molecular weight descriptor
    single_MolWt   = Descriptors.MolWt(mol)
    
    # calculate of the number of rotatable bonds descriptor
    single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
    
    # calculate the aromatic proportion descriptor
    single_AP = getAromaticProportion(mol)

    # Calculate ring count 
    single_RC= Descriptors.RingCount(mol)

    # Calculate TPSA 
    single_TPSA=Descriptors.TPSA(mol)

    # Calculate H Donors  
    single_Hdonors=Lipinski.NumHDonors(mol)

    # Calculate saturated Rings 
    single_SR= Lipinski.NumSaturatedRings(mol) 

    # Calculate Aliphatic rings 
    single_AR =Lipinski.NumAliphaticRings(mol)
    
    # Calculate Hydrogen Acceptors 
    single_HA = Lipinski.NumHAcceptors(mol)

    # Calculate Heteroatoms
    single_Heter = Lipinski.NumHeteroatoms(mol)

    # put the descriptors in a list
    rows = np.array([single_MolLogP, single_MolWt, single_NumRotatableBonds, single_AP,single_RC,single_TPSA,single_Hdonors,single_SR,single_AR,single_HA,single_Heter])
    
    # add the list to a pandas dataframe
    #single_df = pd.DataFrame(single_list).T
    baseData = np.vstack([rows])
    # rename the header columns of the dataframe
    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    #descriptors =np.array(descriptors) 
    #preds=loaded_model.predict(descriptors)
    return model.predict(descriptors)

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        smiles = request.form["smiles"]
    #print(smiles)
    predOUT = predictSingle(smiles, model)
    #predOUT = predOUT +0.20

    return render_template('index.html', prediction_text = "The log S is {}".format(predOUT))
    #return render_template('sub.html',resu= "The log S is {}".format(predOUT))  

if __name__ == "__main__":
    app.run(debug=True, port=5000)

