<template>
  <section class="section">
    <h2 class="section-title">How It Works</h2>
    <p class="section-sub">Five steps from config to trained model.</p>
    <div class="steps">
      <div class="step" v-for="(s, i) in steps" :key="i">
        <div class="step-num">{{ i + 1 }}</div>
        <div>
          <h3>{{ s.title }}</h3>
          <p>{{ s.desc }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const steps = [
  { title: 'Configure', desc: 'Set your API key, dataset path, model, epochs, metrics, and training flags in the GUI.' },
  { title: 'Rent a GPU', desc: 'The orchestrator searches Vast.ai for the cheapest GPU matching your filters and creates an instance.' },
  { title: 'Upload Data', desc: 'Your dataset (images or CSV) is uploaded via SFTP. Flat folders are auto-organized into classes.' },
  { title: 'Train', desc: 'Training runs remotely with live log streaming. Classification uses pretrained CNNs, regression uses MLP.' },
  { title: 'Download Results', desc: 'Model weights, plots, metrics, and Excel report are downloaded. Instance is stopped or destroyed.' },
]
</script>

<style scoped>
.steps{max-width:640px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
.step{display:flex;gap:20px;align-items:flex-start}
.step-num{
  flex-shrink:0;width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1rem;color:#fff;
}
.step h3{font-size:1rem;font-weight:600;margin-bottom:4px}
.step p{color:var(--muted);font-size:.9rem}
</style>
