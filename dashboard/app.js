"use strict";
// All network requests use same-origin FastAPI endpoints. No personal data is saved.
const $ = (id) => document.getElementById(id);
const numericFields = [
  ["age","Age", "years",0,120,1], ["bp","Blood pressure","mmHg",1,400,1],
  ["bgr","Random blood glucose","mg/dL",0.01,2000,"any"],
  ["bu","Blood urea","mg/dL",0.01,1000,"any"], ["sc","Serum creatinine","mg/dL",0.01,100,"any"],
  ["sod","Sodium","mEq/L",0.01,300,"any"], ["pot","Potassium","mEq/L",0.01,20,"any"],
  ["hemo","Hemoglobin","g/dL",0.01,30,"any"], ["pcv","Packed cell volume","%",0.01,100,"any"],
  ["wbcc","White blood cell count","cells/µL",0.01,500000,"any"], ["rbcc","Red blood cell count","million/µL",0.01,20,"any"]
];
const choiceFields = [
  ["sg","Specific gravity",["1.005","1.010","1.015","1.020","1.025"],"urine-fields"],
  ["al","Albumin level",["0","1","2","3","4","5"],"urine-fields"],
  ["su","Sugar level",["0","1","2","3","4","5"],"urine-fields"],
  ["rbc","Red blood cells",["normal","abnormal"],"urine-fields"],
  ["pc","Pus cells",["normal","abnormal"],"urine-fields"],
  ["pcc","Pus cell clumps",["notpresent","present"],"urine-fields"],
  ["ba","Bacteria",["notpresent","present"],"urine-fields"],
  ["htn","Hypertension",["no","yes"],"condition-fields"],
  ["dm","Diabetes mellitus",["no","yes"],"condition-fields"],
  ["cad","Coronary artery disease",["no","yes"],"condition-fields"],
  ["appet","Appetite",["good","poor"],"condition-fields"],
  ["pe","Pedal edema",["no","yes"],"condition-fields"],
  ["ane","Anemia",["no","yes"],"condition-fields"]
];
// Synthetic demonstration values, not patient records, medical advice, or benchmark labels.
const exampleA = {age:50,bp:80,bgr:100,bu:36,sc:1.1,sod:140,pot:4.3,hemo:14,pcv:42,wbcc:7500,rbcc:4.8,sg:"1.020",al:"0",su:"0",rbc:"normal",pc:"normal",pcc:"notpresent",ba:"notpresent",htn:"no",dm:"no",cad:"no",appet:"good",pe:"no",ane:"no"};
const exampleB = {age:60,bp:90,bgr:240,bu:80,sc:3.2,sod:132,pot:5.1,hemo:9.8,pcv:30,wbcc:10000,rbcc:3.9,sg:"1.010",al:"3",su:"2",rbc:"abnormal",pc:"abnormal",pcc:"present",ba:"notpresent",htn:"yes",dm:"yes",cad:"no",appet:"poor",pe:"yes",ane:"yes"};
function addField(container, id, label, elem) {
  const wrap = document.createElement("div"); wrap.className="field";
  const lab = document.createElement("label"); lab.setAttribute("for",id); lab.textContent=label;
  elem.id=id; elem.name=id; elem.required=true; wrap.append(lab,elem); $(container).appendChild(wrap);
}
function createForm() {
  for(const [id,label,unit,min,max,step] of numericFields){
    const input=document.createElement("input");input.type="number";input.min=String(min);input.max=String(max);input.step=String(step);input.placeholder=unit;
    addField("numeric-fields",id,`${label} (${unit})`,input);
  }
  for(const [id,label,values,group] of choiceFields){
    const select=document.createElement("select");const blank=document.createElement("option");blank.value="";blank.textContent="Select value";select.appendChild(blank);
    for(const v of values){const option=document.createElement("option");option.value=v;option.textContent=({yes:"Yes",no:"No",normal:"Normal",abnormal:"Abnormal",present:"Present",notpresent:"Not present",good:"Good",poor:"Poor"})[v] || v;select.appendChild(option)}
    addField(group,id,label,select);
  }
}
function fillExample(values){
  for(const [key,value] of Object.entries(values)) if($(key)) $(key).value=String(value);
  $("form-error").hidden=true;clearResult();
}
function clearResult(){ $("result-empty").hidden=false;$("result-content").hidden=true; }
function setError(text){$("form-error").textContent=text;$("form-error").hidden=false;}
async function getJSON(path,options){
  const response=await fetch(path,{cache:"no-store",...options});
  let payload;try{payload=await response.json()}catch{throw new Error(`Unexpected response from ${path}`)}
  if(!response.ok){const detail=payload?.detail;const readable=Array.isArray(detail)?detail.map(e=>`${(e.loc||[]).slice(-1)[0]||"Input"}: ${e.msg}`).join("; "):(typeof detail==="string"?detail:response.statusText);throw new Error(`${response.status}: ${readable}`)}
  return payload;
}
async function refreshStatus(){
  const [health,ready,info,metrics]=await Promise.allSettled([getJSON("/health"),getJSON("/ready"),getJSON("/model-info"),getJSON("/metrics")]);
  const online=health.status==="fulfilled";
  $("api-indicator").textContent=online?"API online":"API unavailable";$("api-indicator").className=`indicator ${online?"good":"bad"}`;
  $("health").textContent=online?(health.value.status||"Healthy"):"Offline";
  $("readiness").textContent=ready.status==="fulfilled"?"Ready for prediction":"Not ready";
  if(info.status==="fulfilled"){
    const m=info.value;$("version").textContent=m.model_version||"—";$("algorithm").textContent=m.algorithm||"—";
    $("info-algorithm").textContent=m.algorithm||"—";$("features").textContent=String(m.number_of_features??"—");
    const fmt=(v)=>Number.isFinite(Number(v))?`${(Number(v)*100).toFixed(2)}%`:"—";
    $("accuracy").textContent=fmt(m.evaluation_metrics?.accuracy);$("f1").textContent=fmt(m.evaluation_metrics?.f1_ckd);$("auc").textContent=fmt(m.evaluation_metrics?.roc_auc);
  }
  if(metrics.status==="fulfilled"){
    const m=metrics.value;$("requests").textContent=String(m.total_requests);$("successes").textContent=String(m.successful_predictions);
    $("failures").textContent=String(m.failed_requests);$("latency").textContent=`${Number(m.average_response_time_ms).toFixed(1)} ms`;
  }
}
async function submitPrediction(event){
  event.preventDefault();$("form-error").hidden=true;
  const form=$("prediction-form");if(!form.reportValidity())return;
  const payload={};for(const [id] of numericFields)payload[id]=Number($(id).value);
  for(const [id] of choiceFields)payload[id]=$(id).value;
  const submit=$("predict");submit.disabled=true;submit.textContent="Running prediction…";
  try{
    const result=await getJSON("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const ckd=Number(result.probabilities?.CKD),notckd=Number(result.probabilities?.["Not CKD"]);
    if(!Number.isFinite(ckd)||!Number.isFinite(notckd))throw new Error("API returned invalid class probabilities");
    $("result-empty").hidden=true;$("result-content").hidden=false;
    $("prediction-label").textContent=result.prediction;$("prediction-label").className=`prediction-label ${result.prediction==="CKD"?"ckd":""}`;
    $("ckd-pct").textContent=`${(ckd*100).toFixed(2)}%`;$("non-pct").textContent=`${(notckd*100).toFixed(2)}%`;
    $("ckd-bar").style.width=`${Math.min(100,Math.max(0,ckd*100))}%`;
    $("non-bar").style.width=`${Math.min(100,Math.max(0,notckd*100))}%`;
    $("model-note").textContent=result.note||"Model output only; not a medical diagnosis.";
  }catch(error){setError(error.message||"Prediction failed");clearResult();}
  finally{submit.disabled=false;submit.textContent="Generate model prediction";await refreshStatus();}
}
createForm();$("prediction-form").addEventListener("submit",submitPrediction);
$("sample-normal").addEventListener("click",()=>fillExample(exampleA));
$("sample-ckd").addEventListener("click",()=>fillExample(exampleB));
$("clear").addEventListener("click",()=>{$("prediction-form").reset();$("form-error").hidden=true;clearResult()});
$("refresh").addEventListener("click",refreshStatus);refreshStatus();
