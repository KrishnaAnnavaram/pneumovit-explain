# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of pneumovit-explain and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the pneumovit-explain documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **image** | One chest X-ray file. | scan, picture, radiograph (in prose) |
| **patient** | The person ID that the code reads from the file name. | subject, case |
| **train part** | The images that fit the model and fill the neighbour index. | training set (in prose) |
| **val part** | The images that choose hyperparameters, the best epoch, the temperature and the threshold. | validation set, dev set |
| **test part** | The images that the code scores once, at the end. | holdout, evaluation set |
| **probability** | The calibrated pneumonia probability after temperature scaling. | score, confidence |
| **operating threshold** | The probability cut-off, chosen on the val part for the target sensitivity. | cut-point, decision boundary |
| **flag** | The result "above threshold: review for pneumonia" or "below threshold". | diagnosis, prediction (for the flag) |
| **bundle** | The saved final object: classifier, temperature, threshold and neighbour index. | checkpoint (for the bundle) |
| **light classifier** | `light_logreg`: logistic regression on handcrafted X-ray features. | baseline model |
| **deep classifier** | `vit_scratch` or a pretrained timm backbone. | network, net |
| **neighbour index** | The cosine index over embeddings of train-part images of both classes. | retrieval database |
| **saliency map** | The drop of the pneumonia logit when a patch is blurred, for each region. | heat map (alone), attention map |
| **lung zone** | One of six regions: right or left, upper, middle or lower (patient side). | lobe, area |
| **summary** | The plain-language text from the facts, with the banner at the end. | explanation (for the text), rationale |
| **facts** | The measured values that the summary can use. | context, inputs |
| **banner** | The text "NOT A DIAGNOSTIC DEVICE..." at the end of every summary. | disclaimer (in prose) |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **index** | Scan the folder, read the patients and make the patient-aware split. |
| **train** | Fit on the train part, then select and calibrate on the val part. |
| **calibrate** | Fit the temperature on the val part. |
| **evaluate** | Score the test part once and write the report. |
| **explain** | Give the probability, the neighbours, the saliency map and the summary for one image. |
| **query** | Find the most similar train-part images in the neighbour index. |
