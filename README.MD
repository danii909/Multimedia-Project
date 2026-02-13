# 🎬 Video Stabilization System

Sistema di stabilizzazione video basato sulla **stima e compensazione del movimento globale della camera**.  
Il progetto è stato sviluppato nell’ambito del corso di **Multimedia e Laboratorio** e ha come obiettivo lo studio, l’implementazione e il confronto di tecniche classiche di stabilizzazione video.

Il sistema implementa una pipeline modulare che consente il confronto tra:
- **Block Matching**
- **Optical Flow (Lucas–Kanade + RANSAC)**

con valutazione **quantitativa** dei risultati.

---

## 📌 Obiettivi del progetto

Gli obiettivi principali sono:

1. Implementare una pipeline completa di stabilizzazione video basata su:
   - stima del movimento inter-frame
   - estrazione del movimento globale
   - filtraggio temporale della traiettoria
   - compensazione geometrica del moto

2. Confrontare due approcci di stima del movimento:
   - **Block Matching**, coerente con la teoria classica vista a lezione
   - **Optical Flow**, come estensione più robusta per scenari reali

3. Valutare i risultati tramite **metriche oggettive**, evitando valutazioni puramente percettive.

4. Analizzare limiti e punti di forza dei due approcci in diversi scenari di ripresa.

---

## 🧠 Descrizione generale

La stabilizzazione video viene affrontata come un problema di **stima del movimento globale della camera**.  
I movimenti non intenzionali (jitter) sono tipicamente caratterizzati da componenti ad alta frequenza nel tempo, mentre i movimenti intenzionali sono a bassa frequenza.

Il sistema:
1. stima il movimento globale tra frame consecutivi;
2. integra il moto nel tempo per ottenere una traiettoria cumulativa;
3. filtra la traiettoria per rimuovere le componenti ad alta frequenza;
4. applica una trasformazione geometrica inversa per stabilizzare il video.

---

## 🏗️ Architettura della pipeline

La pipeline è suddivisa in moduli indipendenti:

### Moduli principali

- **motion_estimation.py**  
  Stima del moto locale tramite Block Matching.

- **global_motion.py**  
  Stima del movimento globale della camera:
  - aggregazione dei vettori locali (Block Matching)
  - oppure stima tramite Optical Flow + RANSAC.

- **trajectory_smoothing.py**  
  Filtraggio temporale della traiettoria cumulativa.

- **motion_compensation.py**  
  Compensazione del movimento tramite trasformazione affine inversa.

- **video_stabilizer.py**  
  Orchestrazione della pipeline completa.

- **main.py**  
  Script di esecuzione principale.

- **compare_metrics.py**  
  Script per il confronto quantitativo dei risultati.

---

## 🔍 Metodi di stima del movimento

### Block Matching

**Principio**  
Il frame viene suddiviso in blocchi regolari. Per ciascun blocco viene cercata la migliore corrispondenza nel frame successivo minimizzando una metrica di similarità (SAD/MAD).

**Caratteristiche**
- semplice e deterministico
- coerente con la teoria classica di stima del moto
- limitato a spostamenti traslazionali
- sensibile ad ambiguità su regioni poco testurizzate
- poco efficace in presenza di rotazioni

---

### Optical Flow (Lucas–Kanade + RANSAC)

**Principio**  
Vengono rilevate feature distintive (Shi–Tomasi), tracciate nel tempo con Optical Flow di Lucas–Kanade e utilizzate per stimare un modello globale tramite RANSAC.

**Caratteristiche**
- stima sub-pixel del moto
- maggiore robustezza a jitter e rumore
- possibilità di stimare piccole rotazioni
- costo computazionale superiore rispetto al Block Matching

L’Optical Flow non sostituisce il Block Matching, ma ne rappresenta un’estensione per scenari più complessi.

---

## 📈 Traiettoria e filtraggio

Il movimento globale stimato viene integrato nel tempo per ottenere una **traiettoria cumulativa** della camera.

La traiettoria viene filtrata tramite:
- **media mobile**
- **filtro gaussiano**
- **filtro esponenziale (EMA)**

Questo passaggio consente di separare i movimenti non intenzionali da quelli intenzionali ed è direttamente collegabile ai concetti di filtraggio di segnali discreti nel tempo.

---

## 🎯 Compensazione del movimento

La stabilizzazione finale viene ottenuta applicando una **trasformazione affine inversa** a ciascun frame.

Non viene applicato un crop adattivo automatico: eventuali bordi neri sono una conseguenza della compensazione e rappresentano un limite noto del metodo.

---

## 📊 Metriche di valutazione

Durante l’esecuzione vengono calcolate e salvate metriche quantitative in file JSON:

### Metriche di movimento
- RMS degli spostamenti (X, Y)
- RMS dell’angolo di rotazione

### Metriche di stabilità
- riduzione percentuale del jitter (raw → smoothed)
- offset massimo applicato

### Metriche computazionali
- tempo medio per frame
- tempo totale di elaborazione
- FPS effettivi

📌 **Nota**  
I valori numerici dipendono dal video di input e dalla configurazione utilizzata.  
I risultati reali sono contenuti nei file JSON generati dall’esecuzione.

---

## 🚀 Utilizzo

### Stabilizzazione con Block Matching
python main.py --config config/config_block_matching.yaml

### Stabilizzazione con Optical Flow
python main.py --config config/config_optical_flow.yaml

### Confronto quantitativo
python compare_metrics.py data/output/video_stabilizzato_block_matching_metrics.json data/output/video_stabilizzato_optical_flow_metrics.json
