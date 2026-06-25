export const meta = {
  name: 'qml-deep-read',
  description: 'Deep-read 15 QML papers in parallel, extract structured notes for literature review',
  phases: [{ title: 'Read', detail: 'one agent per paper, chunk + read + structured notes' }],
}

const HYP = `RESEARCH HYPOTHESIS: There exist tasks where classical ML methods struggle, but quantum methods—run via simulation with 20+ qubits—may excel.`

// depth: 'deep' => read all chunks; 'skim' => first ~3 chunks (intro+method)
const PAPERS = [
  ['liu2021_rigorous_quantum_speedup', 'deep', 'Liu, Arunachalam, Temme 2021 - provable quantum speedup, discrete-log dataset'],
  ['huang2021_power_of_data', 'deep', 'Huang et al 2021 - Power of data, projected quantum kernel, geometric difference g, engineered datasets up to 30 qubits'],
  ['havlicek2019_quantum_feature_spaces', 'deep', 'Havlicek et al 2019 - quantum feature maps, QSVM & variational classifier'],
  ['peters2021_projected_kernel_hardware', 'deep', 'Peters et al 2021 - projected quantum kernel on 30-qubit hardware'],
  ['abbas2021_power_qnn', 'deep', 'Abbas et al 2021 - effective dimension, power of QNNs'],
  ['kubler2021_inductive_bias_quantum_kernels', 'deep', 'Kubler et al 2021 - inductive bias of quantum kernels, when they help/hurt'],
  ['schuld2021_models_are_kernels', 'deep', 'Schuld 2021 - QML models are kernel methods'],
  ['glick2021_covariant_kernels', 'skim', 'Glick et al 2021 - covariant quantum kernels, group-structured data'],
  ['jerbi2023_beyond_kernel', 'deep', 'Jerbi et al 2023 - QML beyond kernel methods, data re-uploading'],
  ['schuld2021_data_encoding_fourier', 'skim', 'Schuld et al 2021 - data encoding & Fourier expressivity'],
  ['du2020_expressive_power_pqc', 'skim', 'Du et al 2020 - expressive power of PQCs vs classical generative models'],
  ['gao2022_quantum_generative', 'skim', 'Gao et al 2022 - quantum correlations enhance generative models'],
  ['lloyd2020_quantum_embeddings', 'skim', 'Lloyd et al 2020 - quantum embeddings / metric learning'],
  ['jerbi2023_shadows_qml', 'skim', 'Jerbi et al 2023 - shadows of QML, classical deployment'],
  ['tang2019_dequantization_recommendation', 'skim', 'Tang 2019 - dequantization, classical algorithm for recommendation'],
]

const SCHEMA = {
  type: 'object',
  required: ['slug', 'title', 'authors', 'year', 'one_line', 'methodology', 'datasets',
    'baselines', 'metrics', 'key_results', 'qubits', 'classical_struggle', 'code', 'relevance', 'markdown_section'],
  properties: {
    slug: { type: 'string' },
    title: { type: 'string' },
    authors: { type: 'string' },
    year: { type: 'string' },
    one_line: { type: 'string', description: 'one-sentence key contribution' },
    methodology: { type: 'string', description: '2-4 sentences on the method/algorithm' },
    datasets: { type: 'string', description: 'datasets used incl. synthetic/engineered; sizes' },
    baselines: { type: 'string', description: 'classical/quantum baselines compared' },
    metrics: { type: 'string' },
    key_results: { type: 'string', description: 'concrete numbers where available' },
    qubits: { type: 'string', description: 'qubit counts used (sim and/or hardware); note if >=20' },
    classical_struggle: { type: 'string', description: 'does it show a task where classical fails & quantum wins? evidence/conditions/caveats' },
    code: { type: 'string', description: 'code/repo links or framework (Qiskit/PennyLane/TFQ); "none" if absent' },
    relevance: { type: 'string', description: 'relevance to the hypothesis: high/medium/low + why' },
    markdown_section: { type: 'string', description: 'a complete literature-review markdown subsection for this paper, ready to paste' },
  },
}

function prompt(slug, depth, hint) {
  const readInstr = depth === 'deep'
    ? `This is a CORE paper. Run the chunker, check the manifest for chunk count, then read EVERY chunk PDF in papers/pages/ for this paper (chunk_001, chunk_002, ... to the last). Do not skip chunks — methodology and results are in later chunks.`
    : `Run the chunker and read the first 3 chunk PDFs (intro + methods). Skim for the key contribution, datasets, baselines, and any qubit counts / claims of quantum advantage.`
  return `You are a research assistant extracting structured notes from a quantum machine learning paper.

${HYP}

PAPER: papers/${slug}.pdf  (${hint})

STEPS:
1. Activate venv and chunk the PDF (3 pages per chunk):
   source .venv/bin/activate && python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/${slug}.pdf --pages-per-chunk 3
2. ${readInstr}
3. Read the chunk PDFs with the Read tool (they are real PDFs).

Extract the fields in the required schema. Be concrete and quantitative. For 'classical_struggle', critically assess whether the paper genuinely demonstrates a task where classical methods fail and quantum (especially simulable, <=30 qubits) wins — note conditions, whether the advantage is proven/empirical/conjectured, and any dequantization or caveats. The 'markdown_section' must be a polished review subsection starting with '#### ${slug}'.`
}

const results = await parallel(PAPERS.map(([slug, depth, hint]) => () =>
  agent(prompt(slug, depth, hint), { label: `read:${slug}`, phase: 'Read', schema: SCHEMA })
))

const ok = results.filter(Boolean)
log(`Deep-read complete: ${ok.length}/${PAPERS.length} papers`)
return ok
