"""
该部分代码用于从materials project中筛选可用晶体条目，并生成XRD patterns数据集保存在本代码所在路径下的XRD_patterns文件中。
注意！在使用该代码之前，将“Replace with your materials project API key”按要求进行替换。
The code is used to filter the available crystal entries from the materials project,
and generate XRD patterns dataset, which is saved in the xrd_patterns file under the path of this code.
Note! Before using this code, replace with your materials project API key as required.
"""
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import numpy as np
import h5py
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def calc_std_dev(two_theta, tau):
    calculator = xrd.XRDCalculator()
    K = 0.9
    wavelength = calculator.wavelength * 0.1
    theta = np.radians(two_theta/2.)
    beta = (K * wavelength) / (np.cos(theta) * tau)
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2

def add_gaussian_noise_to_xrd(data, mean, std_dev):
    noise2 = np.abs(np.random.normal(mean, std_dev, len(data)))
    noisy_data = data + noise2
    return noisy_data

def normalize_xrd(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [round((item - min_val) / (max_val - min_val) * 100, 3) for item in data]
    return normalized_data

def spectrum(pattern, id, cs, sg, min_angle=0.0, max_angle=180.0, step=0.01):
    angles = list(pattern.keys())
    intensities = list(pattern.values())
    # print(angles,'\n',intensities)
    num = int((max_angle - min_angle) / step + 1)
    steps = np.linspace(min_angle, max_angle, num)
    signals = np.zeros([len(angles), steps.shape[0]])

    for i, ang in enumerate(angles):
        idx = np.argmin(np.abs(ang - steps))
        signals[i, idx] = intensities[i]

    domain_size = 25.0
    step_size = step
    for i in range(signals.shape[0]):
        row = signals[i, :]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size)
        signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size, mode='constant')

    signal = np.sum(signals, axis=0)
    signal = 100 * signal / max(signal)

    pattern2 = np.around(signal[500:9001], decimals=3)#choose 5-90°
    features = add_gaussian_noise_to_xrd(pattern2, mean=0, std_dev=0.6)
    features = normalize_xrd(features)

    labels7 = int(cs) - 1
    labels230 = int(sg) - 1

    save_path = r'./XRD_patterns/'
    name = save_path + id + '.npy'
    if not os.path.exists(name):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        np.save(name, {'data': features, 'crystal system': labels7, 'space group': labels230})
    else:
        print(f'{name} has existed!')

if __name__ == '__main__':
    with MPRester('Replace with your materials project API key') as mpr:
        search_params = {
            # 'material_ids': {'mp-1070'},
            # 'formula': {'Ca'},#'C' Ca
            'deprecated': False,
            'fields': ["material_id", "formula_pretty", "structure",
                       "symmetry", "theoretical", "formation_energy_per_atom"]
        }

        docs = mpr.materials.summary.search(**search_params)
        print(f"Found {len(docs)} materials.\n")

        materials_list = []
        for doc in docs:
            dict = {"material_id": '', "formula_pretty": '', "crystal_system": '', "space_group": '', "theoretical": '',
                    "formation_energy_per_atom": ''}
            for key in dict.keys():
                if key == "crystal_system":
                    dict[key] = str(doc.symmetry.crystal_system)
                elif key == "space_group":
                    dict[key] = doc.symmetry.number
                else:
                    dict[key] = getattr(doc, key)
            materials_list.append(dict)

        # The following is the schema of materials project crystal entries de-duplication protocol.
        selected_materials = {}  # {key1:[{},...,{}], ... , keyN:[{},...,{}]}
        keep_materials = {}  # {key1:[{},...,{}], ... , keyN:[{},...,{}]}
        for material in materials_list:
            formula = material['formula_pretty']
            flag = False
            if formula not in keep_materials.keys():
                selected_materials[formula] = []
                selected_materials[formula].append(material)
                keep_materials[formula] = selected_materials[formula]
                continue
            else:
                for data in keep_materials[formula]:
                    if str(data['crystal_system']) == str(material['crystal_system']) and data['space_group'] == \
                            material['space_group']:
                        if data['theoretical'] and not material['theoretical']:
                            selected_materials[formula].remove(data)
                            selected_materials[formula].append(material)
                        elif not data['theoretical'] and material['theoretical']:
                            flag = False
                        else:
                            if data['formation_energy_per_atom'] is None and material[
                                'formation_energy_per_atom'] is not None:
                                selected_materials[formula].remove(data)
                                selected_materials[formula].append(material)
                            elif data['formation_energy_per_atom'] is not None and material[
                                'formation_energy_per_atom'] is not None:
                                if data['formation_energy_per_atom'] > material['formation_energy_per_atom']:
                                    selected_materials[formula].remove(data)
                                    selected_materials[formula].append(material)
                            else:
                                flag = False
                        flag = False
                        break
                    else:
                        flag = True
                        continue
            if flag:
                selected_materials[formula].append(material)
            keep_materials[formula] = selected_materials[formula]
        # print(selected_materials)

        crystal_system = {'Triclinic': 1, 'Monoclinic': 2, 'Orthorhombic': 3, 'Tetragonal': 4,
                          'Trigonal': 5, 'Rhombohedral': 5, 'Hexagonal': 6, 'Cubic': 7}
        for indexs in selected_materials.values():
            for index in indexs:
                id = str(index['material_id'])
                cs = crystal_system[index['crystal_system']]
                sg = int(index['space_group'])
                try:
                    structure = mpr.get_structure_by_material_id(id)
                    sga = SpacegroupAnalyzer(structure)
                    conventional_structure = sga.get_conventional_standard_structure()

                    xrd_calculator = XRDCalculator(wavelength="CuKa")
                    xrd_pattern = xrd_calculator.get_pattern(conventional_structure)

                    two_theta = np.array(xrd_pattern.x, dtype=np.float64)
                    intensity = np.array(xrd_pattern.y, dtype=np.float64)
                    pattern = {}
                    for i, j in zip(two_theta, intensity):
                        pattern[float(i)] = float(j)

                    spectrum(pattern, id, cs, sg, min_angle=0.0, max_angle=180.0, step=0.01)

                except:
                    print('error：', material_id)

"""
# 下列代码用于可视化本代码最终生成的单条xrd pattern的.npy文件
# The following code is used to visualize the .npy file of the single XRD pattern finally generated by this code

def visualize_xrd(d):
    x = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    o = [1, 500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500]
    plt.plot(d,linewidth=0.5)
    plt.xticks(o,x)
    #plt.savefig('xrd.png')
    plt.show()

loaded_data = np.load(r'mp-xxx.npy', allow_pickle=True).item() # mp-xxx.npy: the .npy file's path
features = loaded_data.get('data')
labels7 = loaded_data.get('crystal system')
labels230 = loaded_data.get('space group')
visualize_xrd(features)
print(labels7)
print(labels230)
print(len(features))
"""

