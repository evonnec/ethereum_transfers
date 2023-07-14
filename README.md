# Ethereum Transfers
Measure Ethereum `ERC-20`` transfer volume with two metrics: (1) count of transfers and (2) sum of transfer amounts  

## set up the environment
`python -m venv ethereum`  
`source ethereum/bin/activate`  

## install requirements
`pip install -r requirements.txt`  

## problem statement
Use the Ethereum Logs data below to find the most transferred ERC-20 token per day.  
  
A list of ERC-20 tokens is also provided in `tokens.csv`.  
  
Specifically, measure transfer volume with two metrics:  
(1) count of transfers  
(2) sum of transfer amounts  
  
The resulting dataset should be two CSVs, one for each metric, each with 3 columns  
  
(day, token address, transfer metric), ordered by transfer metric descending, limit 100.  
  
Data Contents  
  
- Ethereum Decoded Logs (Parquet)
- ERC-20 Token List (csv)


### Examine the data

BLOCK_DATE is transaction date  
`sample_output1 = ["ADDRESS", "BLOCK_DATE", "count_of_transfers"]`  
`sample_output2 = ["ADDRESS", "BLOCK_DATE", "quantity_of_transfers"]`
  
If token addresss exists in tokens.csv, this is included in the metric, else next transaction
Fields `symbol` and `name` are applicable in determining if the token's address from the logs, exist





