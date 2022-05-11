# README #


### What is this repository for? ###

PCa classification based on radiomics features - simple models that base on sklearn classifiers.
[Scripts with experiments](src/scripts) base on .json configuration (see [example](src/scripts/example.py) for very brief documentation)

ISMIS'17: [configuration](src/scripts/ISMIS17.json) , [script](src/scripts/ISMIS17.py)

https://www.researchgate.net/publication/317660491_MRI_imaging_texture_features_in_prostate_lesions_classification


ICBBT'17: [configuration](src/scripts/ICBBT17.json) , [script](src/scripts/ICBBT17.py)

https://www.researchgate.net/publication/317721536_Feature_Extraction_Optimized_For_Prostate_Lesion_Classification

### How do set up the project? ###

* Download ProstateX data
* Enforce correct directory structure

### Directory structure ###

/{project.dir}/data/ProstateX/train/DOI

/{project.dir}/data/ProstateX/train/ktrans

/{project.dir}/data/ProstateX/train/lesion-information/

/{project.dir}/data/ProstateX/train/lesion-information/

/{project.dir}/data/ProstateX/train/lesion-information/ProstateX.csv (if absent generated automatically by [Dataset](src/dataset.py))
\- a dataset file containing data merged from patients mhd files and meta-data files (csv)

/{project.dir}/data/ProstateX/train/lesion-information/ProstateX-DataInfo-Test.docx

/{project.dir}/data/ProstateX/train/lesion-information/ProstateX-Findings.csv

/{project.dir}/data/ProstateX/train/lesion-information/ProstateX-Images.csv

/{project.dir}/data/ProstateX/train/lesion-information/ProstateX-Images-KTrans.csv

/{project.dir}/data/ProstateX/train/screenshots (optional)
