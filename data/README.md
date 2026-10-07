# Data

Git does not store X-ray images in this repository. Put the dataset in `data/` (git ignores
everything here except this file), or point `PNEUMOVIT_DATA_DIR` at it.

## Source

| Item | Value |
|---|---|
| Name | Chest X-Ray Images (Pneumonia) |
| URL | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Original | Kermany et al., Cell 2018. Mendeley Data: https://data.mendeley.com/datasets/rscbjbr9sj |
| License | CC BY 4.0 (Mendeley Data). Give credit to the authors |
| Content | 5,863 pediatric anterior-posterior chest X-rays (ages 1 to 5, one centre), NORMAL or PNEUMONIA (bacterial or viral) |
| Size | About 1.2 GB |

## Download

1. Sign in to Kaggle and accept the dataset terms.
2. Download the archive and extract it so that `data/chest_xray/train`, `val` and `test` exist.
3. Run `pneumovit-explain index --data-dir data/chest_xray`.

With the Kaggle CLI: `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip`.

## Expected layout

```
data/chest_xray/
  train/NORMAL  train/PNEUMONIA
  val/NORMAL    val/PNEUMONIA      only 16 images: the code pools all parts and splits again by patient
  test/NORMAL   test/PNEUMONIA
```

## Patient IDs

The code reads the patient from the file name:

| File name | Patient | Subtype |
|---|---|---|
| `person1_bacteria_1.jpeg` | `P:person1` | bacteria |
| `person1_virus_6.jpeg` | `P:person1` | virus |
| `IM-0115-0001.jpeg` | `N:IM-0115` | normal |
| `NORMAL2-IM-1427-0001.jpeg` | `N:NORMAL2-IM-1427` | normal |
| other names | the file itself | one patient per file (warning) |

## Privacy

Do not put identifiable patient images in the app. The app keeps uploads in memory and does not
write them to disk. The LLM summary sends no image unless `PNEUMOVIT_LLM_SEND_IMAGE=true`.

## Offline data

The tests and the demo use synthetic X-ray-like images. They need no download:

```bash
pneumovit-explain synth --out data/synthetic/chest_xray --patients 300
```

The generator also writes `metadata.csv` with the true zone of each focal finding.
