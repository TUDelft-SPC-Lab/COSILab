
### Model-human answer similarity

|        | **metric** | **field**   | **source**        | **n** | **mean_similarity** | **std**  |
| ------ | ---------- | ----------- | ----------------- | ----- | ------------------- | -------- |
| **0**  | cosine     | description | model             | 358   | 0.302909            | 0.112388 |
| **1**  | cosine     | description | persona           | 424   | 0.321919            | 0.119854 |
| **2**  | cosine     | description | persona_annotator | 1182  | 0.307602            | 0.112986 |
| **3**  | cosine     | explanation | model             | 364   | 0.319638            | 0.092202 |
| **4**  | cosine     | explanation | persona           | 424   | 0.340755            | 0.085296 |
| **5**  | cosine     | explanation | persona_annotator | 1185  | 0.335519            | 0.091636 |
| **6**  | hts        | description | model             | 358   | 0.289653            | 0.100173 |
| **7**  | hts        | description | persona           | 424   | 0.306327            | 0.106711 |
| **8**  | hts        | description | persona_annotator | 1182  | 0.293799            | 0.101424 |
| **9**  | hts        | explanation | model             | 364   | 0.305927            | 0.081495 |
| **10** | hts        | explanation | persona           | 424   | 0.325134            | 0.075587 |
| **11** | hts        | explanation | persona_annotator | 1185  | 0.320154            | 0.080699 |
| **12** | overlap    | description | model             | 358   | 0.185287            | 0.082288 |
| **13** | overlap    | description | persona           | 424   | 0.199596            | 0.088007 |
| **14** | overlap    | description | persona_annotator | 1182  | 0.188768            | 0.081517 |
| **15** | overlap    | explanation | model             | 364   | 0.195253            | 0.068532 |
| **16** | overlap    | explanation | persona           | 424   | 0.209999            | 0.062959 |
| **17** | overlap    | explanation | persona_annotator | 1185  | 0.206693            | 0.068570 |

### persona specific distribution to the whole no persona distribution for model answers

|      | field       | metric  | n_persona_pairs | within_persona_similarity | within_persona_std | n_model_pairs | general_model_similarity | general_model_std | normalized_delta | normalized_ratio | lower_variance |
| ---- | ----------- | ------- | --------------- | ------------------------- | ------------------ | ------------- | ------------------------ | ----------------- | ---------------- | ---------------- | -------------- |
| 0    | description | cosine  | 36056           | 0.353195                  | 0.165954           | 1367031       | 0.324973                 | 0.142845          | 0.028222         | 1.086844         | model          |
| 1    | description | overlap | 36056           | 0.228253                  | 0.138638           | 1367031       | 0.203289                 | 0.109918          | 0.024963         | 1.122795         | model          |
| 2    | description | hts     | 36056           | 0.330756                  | 0.139276           | 1367031       | 0.308162                 | 0.124702          | 0.022594         | 1.073317         | model          |
| 3    | explanation | cosine  | 36553           | 0.369652                  | 0.125907           | 1447551       | 0.321270                 | 0.116410          | 0.048382         | 1.150596         | model          |
| 4    | explanation | overlap | 36553           | 0.234524                  | 0.101813           | 1447551       | 0.197347                 | 0.086762          | 0.037177         | 1.188383         | model          |
| 5    | explanation | hts     | 36553           | 0.348750                  | 0.106885           | 1447551       | 0.306848                 |                   |                  |                  |                |