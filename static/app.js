let shrimps=[];
async function j(url,opt){const r=await fetch(url,opt);if(!r.ok) throw new Error(await r.text()); return r.json();}
async function loadConfig(){const c=await j('/api/config');document.getElementById('scaleChip').textContent=`SCALE: ${Number(c.SCALE).toFixed(4)} mm/px`;}
async function loadFiles(){const files=await j('/api/files/input');document.getElementById('files').innerHTML=files.map(f=>`<li>${f.filename} (${f.size})</li>`).join('');}
async function loadRuns(){const runs=await j('/api/results/runs');const sel=document.getElementById('runSel');sel.innerHTML=runs.map(r=>`<option>${r}</option>`).join(''); if(runs[0]) await loadSources();}
async function loadSources(){const run=document.getElementById('runSel').value;const src=await j(`/api/results/sources?run_dir=${encodeURIComponent(run)}`);const sel=document.getElementById('srcSel');sel.innerHTML='<option value="">ALL</option>'+src.map(s=>`<option>${s}</option>`).join('');}
async function loadRows(){const run=runSel.value, src=srcSel.value;shrimps=await j(`/api/results/shrimps?run_dir=${encodeURIComponent(run)}&source=${encodeURIComponent(src)}`);rows.innerHTML=shrimps.map((s,i)=>`<tr><td>${s.track_id}</td><td>${s.frame_idx}</td><td>${s.pixel_length.toFixed(1)}</td><td>${s.real_length_mm.toFixed(2)}</td><td><input type='number' min='0.1' step='0.1' data-i='${i}'></td><td class='sc'>--</td></tr>`).join('');calPanel.hidden=false;}
uploadBtn.onclick=async()=>{const fd=new FormData();[...fileInput.files].forEach(f=>fd.append('files',f));await j('/api/files/upload',{method:'POST',body:fd});await loadFiles();};
runBtn.onclick=()=>j('/api/pipeline/run',{method:'POST'});
runSel.onchange=loadSources; loadBtn.onclick=loadRows;
calcBtn.onclick=async()=>{const entries=[];[...rows.querySelectorAll('input')].forEach(inp=>{const v=parseFloat(inp.value);if(v>0){const s=shrimps[+inp.dataset.i];entries.push({source_stem:s.source_stem,track_id:s.track_id,real_length_mm:v});}});if(!entries.length)return;
const run=runSel.value;const res=await j(`/api/calibrate?run_dir=${encodeURIComponent(run)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({entries})});
[...rows.querySelectorAll('tr')].forEach((tr,i)=>{const idx=entries.findIndex(e=>e.track_id===shrimps[i].track_id&&e.source_stem===shrimps[i].source_stem);if(idx>=0)tr.querySelector('.sc').textContent=res.scales_detail[idx].toFixed(6)});
calOut.textContent=JSON.stringify(res,null,2);await loadConfig();};
loadConfig();loadFiles();loadRuns();
