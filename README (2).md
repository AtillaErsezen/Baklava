# ML Factory

Claude (agent) veriyi profiller → deney planlar → Modal'da paralel CV ile eğitir → kazananı seçer → rapor yazar.

## Kurulum (sabah ilk 30 dk)

```bash
pip install -r requirements.txt
modal setup                          # tarayıcıdan login
export ANTHROPIC_API_KEY=sk-ant-...

# 1) Agent'sız smoke test — backend çalışıyor mu? (en büyük risk, önce bunu bitir)
modal run modal_train.py --csv data/titanic.csv --target Survived

# 2) Deploy (agent deploy edilmiş fonksiyonları çağırır)
modal deploy modal_train.py

# 3) Agent
python agent.py data/titanic.csv --target Survived
python agent.py data/houses.csv --target SalePrice --task regression
```

Çıktılar `runs/<run_id>_events.jsonl`, `_results.json`, `_report.md`. Model dosyası Modal volume'da: `/models/<run_id>/<name>.joblib`.

## Dosyalar

- `modal_train.py` – model menüsü, preprocessing, CV, `train_candidate` (bir aday), `fit_final` (kazanan). Claude kod yazmaz; sadece menüden model + parametre seçer.
- `agent.py` – profil, 5 tool, agent döngüsü, event stream. Bütçe: 3 tur × 8 aday.

## Canlı UI (Supabase)

```sql
create table events (
  id bigserial primary key,
  run_id text, ts double precision, kind text, payload jsonb
);
alter publication supabase_realtime add table events;
```

`SUPABASE_URL` ve `SUPABASE_KEY` set edilirse agent her adımı buraya yazar. Lovable UI `events` tablosuna realtime subscribe olup `kind`'a göre render eder:
`run_start`, `thought`, `tool_call`, `round_start`, `leaderboard`, `final_model`, `report`.

**Replay modu:** Demo çökerse UI'a önceden koşturulmuş `runs/*_events.jsonl` dosyasını `ts` sırasıyla oynat.

## Sesli özet

`report.spoken_summary` → ElevenLabs TTS → UI'da play butonu.

## Sabah checklist

- [ ] 2-3 dataset seç ve `data/`'ya koy (bir sınıflandırma, bir regresyon, biri içine bilerek leakage kolonu koyulmuş — agent'ın yakalaması iyi demo anı)
- [ ] Smoke test geçti
- [ ] Deploy + ilk agent run
- [ ] Demo dataset'lerini bir kez koştur, events'i sakla (replay)
