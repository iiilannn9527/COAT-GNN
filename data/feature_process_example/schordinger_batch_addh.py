from schrodinger.test import mmshare_data_file
from schrodinger.structutils import analyze
from schrodinger.structutils import build
from schrodinger import structure
import os
sequence_file = "example.txt"
file_title = sequence_file.split(".")[0]
work_dir = f"structure_{file_title}"
out_path = f"structure_{file_title}_schordinger"
os.makedirs(out_path, exist_ok=True)
#os.chdir(out_path)
file_list = os.listdir(work_dir)
for file_name in file_list:
    if file_name.endswith(".pdb"):
        # print(file_name)
        modified_fname = f'{out_path}/modified_{file_name}'
        if not os.path.exists(f"{out_path}\\{modified_fname}"):
            fname = os.path.join(work_dir,file_name)
            st = structure.StructureReader.read(fname)
            gly_list = []
            for at in st.atom:
                if at.pdbname == ' CA ' and at.pdbres == "GLY ":
                    gly_list.append(at.index)
            build.add_hydrogens(st,'All-atom with No-Lp', gly_list)
            rm_list = []
            for at in st.atom:
                if at.pdbname == ' CA ' and at.pdbres == "GLY ":
                    bonds = list(at.bond)
                    if len(bonds) != 4:
                        raise f"{file_name} and {at.resnum} get erro in gly addh"
                    for bond in bonds:
                        if bond.atom2.pdbname == " HA2":
                                rm_list.append(bond.atom2.index)
                                break
            st.deleteAtoms(rm_list)                
            for at in st.atom:
                if at.pdbname == ' CA ' and at.pdbres == "GLY ":
                    bonds = list(at.bond)
                    if len(bonds) != 3:
                        raise f"{file_name} and {at.resnum} get erro in gly addh"
            side_atom = analyze.evaluate_asl(st, 'sidechain')
            build.add_hydrogens(st,'All-atom with No-Lp', side_atom)

            
            # Now write our modified structure to disk.
            with structure.StructureWriter(modified_fname) as writer:
                writer.append(st)


