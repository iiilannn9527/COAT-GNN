# COAT-GNN Protein Feature Generation Pipeline

## Description
This repository provides scripts to generate protein features for COAT-GNN, including structural processing, feature generation, and feature extraction.

## Requirements

### Software and Databases

1. **[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)**

2. **[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)**

3. **[DSSP](https://github.com/cmbi/dssp)**
   - Alternative installation:
     ```bash
     conda install -c ostrokach dssp
     ```

4. **[ESMFold](https://github.com/facebookresearch/esm)**

5. **[Schrödinger Suite](https://www.schrodinger.com/)**

## Pipeline Steps
### **Step 1: Generate Protein Structure**
- **Input**: Protein sequence file (example.txt).
- **Output**: PDB structure files (ESMFold-predicted).
- **Command**: python generate_structure_for_ppis.py
### **Step 2: Process Structures with Schrödinger**
- **Input**: PDB files from Step 1 or crystal structures.
- **Output**: Hydrogen-added structures.
- **Command**: run schordinger_batch_addh.py
### **Step 3: Feature Extraction**
- **Input**: PDB files and protein sequence file.
- **Output**: Feature files.
- **Command**: python feature_process_for_ppis.py
### **Step 4: Merge Features**
- **Input**: Individual feature files from Step 3.
- **Output**: Merged feature matrix.
- **Command**: python merge_feature_for_ppis.py
