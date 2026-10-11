# Data Processing

## 1. Build Dataset

Run:

* python -m data.build_dataset  

This will generate:

 data/train.json  
 data/val.json  
 data/test.json  
 data/data_splits_report.txt  

* Important 

Before running, delete existing JSON files (if have):

data/train.json  
data/val.json  
data/test.json  

Otherwise, the script will reuse old data instead of regenerating.



## 2. Additional Analysis

Run:

* python -m data.domain_analysis

Output:

 data/analysis/domain_analysis_report.txt  
 data/analysis/domain_analysis_results.json