# Cho2017 Motor Imagery EEG Karşılaştırması: CSP + LDA vs EEGNet vs ATCNet

Yeniden üretilebilir bir BCI araştırma projesi. Cho ve ark. (2017) GigaDB motor
imgeleme veri kümesinde **sol el vs sağ el** motor imgelemeyi, üç yöntemle —
geleneksel **CSP + LDA**, kompakt CNN **EEGNet** ve dikkat + zamansal evrişim
modeli **ATCNet** — *aynı koşullarda* karşılaştırır.

> **Önemli:** Bu depo tüm kodu, defterleri ve iskeleyi içerir; ancak **eğitim
> defterleri çalıştırılmamıştır** ve `results/` klasörlerinde **hiçbir model
> sonucu yoktur**. Sonuçlar yalnızca ilgili defteri siz çalıştırdığınızda oluşur.
> Boş `results/` klasörleri deney sonucu değildir.

---

## 1. Projenin amacı

Derin öğrenmenin geleneksel bir BCI hattına göre motor imgelemeyi *daha doğru ve
daha tutarlı* sınıflandırıp sınıflandıramayacağını; eğer ediyorsa hangi maliyetle
(parametre, eğitim süresi, çıkarım gecikmesi, yorumlanabilirlik) ettiğini ölçmek.
Proje, "AI mutlaka kazanır" varsayımını **yapmaz**: CSP + LDA güçlü bir temel
hattır; amaç doğruluk/karmaşıklık ödünleşmesini dürüstçe ölçmektir.

## 2. Bilimsel motivasyon

Motor imgeleme, sensorimotor ritimlerde (mu 8–13 Hz, beta 13–30 Hz) kontralateral
güç azalması/artışı (ERD/ERS) üretir. CSP + LDA bu uzaysal varyans desenlerini
açıkça modeller; EEGNet/ATCNet ise uçtan uca öğrenir. Performans katılımcılar
arasında çok değişkendir; bu yüzden **denek-bazlı** ve **eşli istatistiksel**
karşılaştırma yapılır.

## 3. Veri kümesi

- 52 sağlıklı katılımcı (`s01`–`s52`), kayıt başına tek oturum.
- 64 EEG + 4 EMG = 68 kanal, 512 Hz.
- Sınıflar: sol el / sağ el imgeleme; sınıf başına 100 veya 120 deneme; 5 veya 6 koşu.
- `s29` ve `s34` orijinal makalede >%90 EMG-korelasyonlu denemeler nedeniyle
  dışlanmıştır; **ana karşılaştırmada dışlanır**, **EDA'da incelenir**.
- EMG **asla** sınıflandırma girdisi değildir; yalnızca artefakt analizi içindir.

`.mat` dosya yapısı (doğrulanmış): üst düzey `eeg` struct'ı; `imagery_left/right`
(68×T sürekli), ikili `imagery_event` işaretçisi, `frame=[-2000,5000]` ms,
sınıfa-özel/bir-tabanlı `bad_trial_indices`. Ayrıntılar:
`results/eda/reports/mat_structure_report.md` ve
`data/metadata/mat_layout_resolution.json`.

## 4. Veri kümesi atfı

> Cho, H., Ahn, M., Ahn, S., Kwon, M., & Jun, S. C. (2017). *EEG datasets for
> motor imagery brain–computer interface.* GigaScience, 6(7), gix034.
> https://doi.org/10.1093/gigascience/gix034 — Veri: https://doi.org/10.5524/100295

## 5. Modeller

- **CSP + LDA** — MNE `CSP` (log-varyans uzaysal filtreler) + scikit-learn LDA.
- **EEGNet** — Braindecode `EEGNetv4` (zamansal + derinlemesine uzaysal + ayrılabilir evrişim).
- **ATCNet** — **varsayılan: Braindecode `ATCNet`**; konv. blok + kayan pencereler +
  çok-başlı dikkat + TCN. Braindecode kullanılamazsa yalnızca **açıkça
  etkinleştirildiğinde** yerel bir PyTorch uygulaması (`LocalATCNet`) devreye girer
  ve sonuçları ayrı etiketlenir (`model_source="local_fallback"`).

Tüm modeller ham **logit** üretir (içeride softmax yok); `CrossEntropyLoss` ile eğitilir.

## 6. Dizin yapısı

```
bci/
├── configs/            base.yaml + model yaml'leri
├── data/
│   ├── raw/cho2017/    sNN.mat (git'e eklenmez)
│   ├── processed/common/ sNN.npz (önbellek)
│   ├── manifests/      file/trial/split/excluded *.csv
│   └── metadata/       channel_names / dataset_summary / preprocessing_summary / mat_layout_resolution
├── notebooks/          00_eda .. 04_final_model_comparison
├── src/cho2017_benchmark/  paths, config, reproducibility, data/, models/, training/, evaluation/, reporting/
├── results/            eda / csp_lda / eegnet / atcnet / comparison
├── scripts/            download / validate / prepare / create_splits / print_project_status
└── tests/              birim ve şekil testleri
```

## 7. Kurulum

```bash
python -m pip install -e .
# veya sabitlenmiş sürümlerle:
python -m pip install -r requirements.txt
# conda:
conda env create -f environment.yml && conda activate cho2017-benchmark
```

Python 3.11 referans sürümdür. `requirements.txt`/`environment.yml` sabitlenmiş,
karşılıklı uyumlu sürümleri belgeler. ATCNet için `braindecode >= 1.1` gerekir.

## 8. Veri indirme

```bash
python scripts/download_dataset.py                 # tüm 52 denek (idempotent, disk-güvenli)
python scripts/download_dataset.py --inspection    # yalnızca s01,s20,s29,s33,s34,s52
python scripts/download_dataset.py --force         # yeniden indir
```

İndirici: disk alanını kontrol eder, `*.part` ile devam ettirir, üstel geri çekilme
ile yeniden dener, başlık doğrulamasından sonra atomik olarak yeniden adlandırır.
`data/manifests/file_manifest.csv` üretilir.

## 9. Veri doğrulama

```bash
python scripts/validate_dataset.py
```

Her dosyayı derinlemesine doğrular (yüklenir mi, `eeg` struct, srate, kanal sayısı,
olay sayısı) ve `mat_structure_report.md` + `mat_layout_resolution.json` üretir.

## 10. Veri hazırlama

```bash
python scripts/prepare_dataset.py      # data/processed/common/sNN.npz üretir
python scripts/create_splits.py        # trial/split/excluded manifestoları + sızıntı kontrolü
```

`prepare_dataset.py`, **doğrulanmış** `mat_layout_resolution.json` olmadan
çalışmayı reddeder. Kanonik ön-işleme (epok başına): CAR → sıfır-faz 8–30 Hz
Butterworth → 512→256 Hz yeniden örnekleme → 0.5–2.5 s kırpma → **64×512**.

## 11. Defter çalıştırma sırası

```
1. Ortamı kur
2. Veriyi indir
3. Veriyi doğrula
4. Epokları ve manifestoları hazırla
5. 00_exploratory_data_analysis.ipynb
6. 01_traditional_csp_lda.ipynb
7. 02_eegnet.ipynb
8. 03_atcnet.ipynb
9. 04_final_model_comparison.ipynb
```

Her defter çekirdek baştan yeniden başlatılabilir; src'den fonksiyon çağırır;
çıktıları kaydeder; "Limitations" ve "Generated Files" bölümleri içerir.

## 12. Quick mode (hızlı mod)

Her defterin başında `QUICK_MODE = True` yalnızca **hata ayıklama** içindir ve
küçük bir denek alt kümesi (`[1,2,3,4,5]`) kullanır. Defterler şu uyarıyı gösterir:

> **QUICK_MODE sonuçları nihai bilimsel sonuç olarak kullanılamaz.**

## 13. Full mode (tam mod)

`QUICK_MODE = False` (varsayılan) 50 uygun katılımcının tamamını kullanır
(`s29`/`s34` hariç). Nöral modeller için tohumlar tam modda `[42, 43, 44]`'tür
(`configs/*.yaml` içinde `seeds`).

## 14. GPU / CPU kullanımı

`device: auto` CUDA varsa GPU, yoksa CPU kullanır. Karışık hassasiyet yalnızca
CUDA'da etkinleşir. Gecikme ölçümü cihazı raporlar; **GPU nöral çıkarımı CPU CSP
ile cihaz farkı belirtilmeden karşılaştırılmaz**.

## 15. Çıktı konumları

- EDA: `results/eda/{figures,tables,reports}`
- Modeller: `results/{csp_lda,eegnet,atcnet}/{tables,predictions,metrics,figures,checkpoints,histories,configs,logs}`
- Karşılaştırma: `results/comparison/{tables,figures,reports}`
- Nöral kontrol noktaları: `checkpoints/<sid>/seed_<seed>/best.pt`; geçmişler:
  `histories/<sid>_seed_<seed>.csv`.

## 16. Yeniden üretilebilirlik

Tek bir `set_seed(42)` Python/NumPy/PyTorch'u (CPU+CUDA) tohumlar ve mümkün olduğunda
deterministik algoritmalar ister (katı determinizm bazı işlemlerde yavaşlatabilir).
Split, ön-işleme ve model yapılandırmaları için deterministik hash'ler üretilir.
Her sonuç klasörüne ortam bilgisi (`environment.json`) kaydedilir.

## 17. Sızıntı önleme kuralları

- Tek paylaşılan split manifestosu; üç deney de aynısını kullanır.
- Test verisi CSP fitleme, normalizasyon, hiperparametre/eşik/erken durdurma
  seçimi için **asla** kullanılmaz.
- Öğrenilen ön-işleme yalnızca eğitimde fit edilir; nöral normalizasyon
  istatistikleri yalnızca eğitim denemelerinden hesaplanıp dondurulur.
- Sıfır-faz filtreleme **epok başına** uygulanır (bütün kayda değil), böylece
  filtre yayılması split sınırlarını aşamaz.
- Dışlanan denemeler bir split'e asla girmez; aday seçimi yalnızca validation'da
  `tuning_seed=42` ile yapılır; reddedilen adaylar test'e dokunmaz.

## 18. Beklenen sınırlılıklar

1. Katılımcı başına tek oturum → gerçek oturumlar-arası genelleme değildir.
2. Ana deney denek-bağımlıdır.
3. Açık koşu etiketi yoktur; koşular protokolden (kronolojik sıra varsayımı) çıkarılır.
4. MI performansı katılımcılar arasında çok değişir.
5. EMG-bazlı denek dışlama tüm artefakt sizintisini gidermez.
6. Çevrimdışı test doğruluğu çevrimiçi BCI kullanılabilirliğine eşit değildir.
7. İstatistiksel anlamlılık pratik/klinik değeri garanti etmez.
8. Sınırlı veriyle derin model aşırı öğrenebilir; CSP + LDA bazı denekler için rekabetçi kalabilir.
9. Hiçbir sonuç, defterler gerçekten çalıştırılmadan iddia edilemez.

## 19. Atıflar (modeller)

- **EEGNet:** Lawhern et al., *EEGNet: a compact CNN for EEG-based BCIs*, J. Neural Eng. 2018.
- **ATCNet:** Altaheri et al., *Physics-informed attention temporal convolutional
  network for EEG-based motor imagery classification*, IEEE TII 2023,
  doi:10.1109/TII.2022.3197419.
- **Braindecode:** Schirrmeister et al., 2017; https://braindecode.org
- **MOABB:** Jayaram & Barachant, 2018 (yalnızca metadata/indirme yedeği için).

## 20. Sorun giderme

- **`braindecode` bulunamadı:** `pip install braindecode` (ATCNet varsayılanı için gerekli).
  Şekil testleri yoksa atlanır.
- **`prepare` reddediyor:** önce `validate_dataset.py` ile `mat_layout_resolution.json` üretin.
- **`make` yok (Windows):** Makefile hedefleri yerine `python scripts/...` komutlarını doğrudan çalıştırın.
- **Disk yetersiz:** indirici durur ve gereken/boş alanı raporlar; `--inspection` ile alt küme indirin.
- **CUDA OOM:** `configs/*.yaml` içinde `batch_size`'ı düşürün veya `device: cpu` yapın.
- **parquet hatası:** `pip install pyarrow`.

---

## Hızlı komut özeti

```bash
python -m pip install -e .            # kurulum
python scripts/download_dataset.py    # indir
python scripts/validate_dataset.py    # doğrula + layout çöz
python scripts/prepare_dataset.py     # işlenmiş önbellek
python scripts/create_splits.py       # manifestolar + sızıntı kontrolü
python scripts/print_project_status.py
pytest -q                             # testler
```

Makefile hedefleri: `install`, `download`, `validate-data`, `prepare-data`,
`splits`, `test`, `clean-cache`, `status`.
