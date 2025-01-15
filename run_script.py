import pandas as pd
import os
from pymatgen.core.structure import Structure



def find_csv():
    return next(f for f in os.listdir('.') if f.endswith('.csv'))

# Read MOF names from CSV file
csv_file = find_csv()  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

def read_cif_unit_cell(cif_file_path):
    try:
        # Read the CIF file
        # structure = Structure.from_file(cif_file_path)
        
        # # Get the unit cell parameters
        # lattice = structure.lattice
        # a, b, c = lattice.abc
        a, b, c = Structure.from_file(cif_file_path).lattice.abc
        
        print(f"Unit Cell Parameters:")
        print(f"a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
        return a, b, c
    except Exception as e:
        print(f"Error reading CIF file: {e}")

def get_multiplication_number_a(a):
    target_a = a
    x = 1
    while target_a < 27:
        x += 1
        target_a = a * x
    return x
def get_multiplication_number_b(b):
    target_b = b
    y = 1
    while target_b < 27:
        y += 1
        target_b = b * y
    return y
def get_multiplication_number_c(c):
    target_c = c
    z = 1
    while target_c < 27:
        z += 1
        target_c = c * z
    return z


# Loop through each MOF in the list
for index, row in df.iterrows():
    MOF_NAME = row['Name']
    HELIUM_VOID_FRACTION = row['VF ']
    cif_file = MOF_NAME + ".cif"
    a, b, c = read_cif_unit_cell(cif_file)
    X = get_multiplication_number_a(a)
    Y = get_multiplication_number_b(b)
    Z = get_multiplication_number_c(c)

    Temperatures = [160, 223, 295]
    for T in Temperatures: 
        input_file = "simulation.input"

        with open(input_file, 'w') as f:
            f.write(f"""\
#########General block#################
SimulationType                MC
NumberOfCycles                2000
NumberOfInitializationCycles  2000
NumberOfEquilibrationCycles   2000
RestartFile                   no
Movies                        no
PrintEvery                    500

#########Force field##################
Forcefield                    Local
UseChargesFromCIFFile         yes


#########Structure block##############
Framework 0
FrameworkName {MOF_NAME}
UnitCells {X} {Y} {Z}
HeliumVoidFraction {HELIUM_VOID_FRACTION}
ExternalTemperature {T}
ExternalPressure 100000 500000 1000000 3500000 6000000 10000000 


#########Molecule block###############
Component 0 MoleculeName             hydrogen
            MoleculeDefinition       Local
            TranslationProbability   0.5
            RotationProbability      0.5
            SwapProbability          1
            ReinsertionProbability   0.5 
            CreateNumberOfMolecules  0
""")

        # Run RASPA simulation
        print(f"Running RASPA simulation for {MOF_NAME}...")
        os.system(f"bash run")


# os.system("rm simulation.input")
