<template>
  <section class="section">
    <h2 class="section-title">Architecture</h2>
    <p class="section-sub">Modular Python codebase — each file has a single responsibility.</p>
    <div class="modules">
      <div class="mod" v-for="m in modules" :key="m.file">
        <code>{{ m.file }}</code>
        <span>{{ m.role }}</span>
      </div>
    </div>
  </section>
</template>

<script setup>
const modules = [
  { file: 'gui.py', role: 'Desktop GUI — config panel, log viewer, SSH console' },
  { file: 'config.py', role: 'Dataclass holding all experiment parameters' },
  { file: 'orchestrator.py', role: '5-step pipeline controller' },
  { file: 'vast_api.py', role: 'Vast.ai SDK wrapper — search, create, stop, destroy' },
  { file: 'ssh_manager.py', role: 'Paramiko SSH/SFTP — exec, upload, download' },
  { file: 'train.py', role: 'Remote training script (classification + regression)' },
]
</script>

<style scoped>
.modules{max-width:640px;margin:0 auto;display:flex;flex-direction:column;gap:10px}
.mod{
  display:flex;align-items:center;gap:16px;
  padding:14px 20px;border-radius:10px;
  background:var(--bg2);border:1px solid var(--bg3);
}
.mod code{
  flex-shrink:0;min-width:140px;
  font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:.85rem;color:var(--accent3);font-weight:500;
}
.mod span{color:var(--muted);font-size:.88rem}
</style>
