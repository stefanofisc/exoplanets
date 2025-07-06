# IT Guida Configurazione del Dataset

Questo file di configurazione consente di:

1. Creare uno **split training-test da file CSV**, e salvarlo in formato CSV;
2. Creare uno **split training-test da file CSV**, e salvarlo in formato tensoriale;
3. **Caricare direttamente dati pre-processati** in formato tensoriale (`.pt`)

Si veda sezione 'Example usage' per un esempio di utilizzo relativo ai punti (2) e (3)

# EN Dataset Configuration Guide.

This configuration file allows you to:

1. Create a **split training-test from CSV file**, and save it in CSV format;
2. Create a **split training-test from CSV file**, and save it in tensor format;
3. **Directly load pre-processed data** in tensor format (`.pt`).

See section ‘Example usage’ for an example of usage related to (2) and (3)

---
## Example usage
2. Creating training-test tensors from scratch
```yaml
dataset_filename: 'tess_tey2023_multiclass_ycategorical.csv'
mapping:
  B: 0
  E: 1
  J: 2
test_size: 0.2
dataset_splitting: True
dataset_save_split_csv: False
dataset_save_split_tensors: True
```
3. Loading training-test tensors
```yaml
dataset_filename: 'tess_tey2023_multiclass_ycategorical.csv'
load_tensors: True
catalog_name: 'tess_tey23'
```

