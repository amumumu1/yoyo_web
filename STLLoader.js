// STLLoader.js — three.js r152 compatible
THREE.STLLoader=function(e){this.manager=void 0!==e?e:THREE.DefaultLoadingManager};
THREE.STLLoader.prototype={constructor:THREE.STLLoader,load:function(e,t,r,o){
var n=this,i=new THREE.FileLoader(n.manager);
i.setPath(n.path),i.setResponseType("arraybuffer"),i.load(e,function(e){
t(n.parse(e))
},r,o)},setPath:function(e){return this.path=e,this},parse:function(e){
var t=function(e){
var t=new DataView(e),r=80,o=t.getUint32(r,true);r+=4;
var n=new Float32Array(9*o),i=0;for(let a=0;a<o;a++){
r+=12;
for(let e=0;e<3;e++){
let o=t.getFloat32(r,true);r+=4;
let s=t.getFloat32(r,true);r+=4;
let f=t.getFloat32(r,true);r+=4;
n[i]=o,n[i+1]=s,n[i+2]=f,i+=3}
r+=2}
return n
};
var r;
if("string"==typeof e){
if(e.startsWith("solid"))return this.parseASCII(e)}else{
if(80!==new Uint8Array(e,0,80).byteLength)return new THREE.BufferGeometry;
r=t(e)}
var o=new THREE.BufferGeometry;
return o.setAttribute("position",new THREE.BufferAttribute(r,3)),o.computeVertexNormals(),o}};
THREE.STLLoader.prototype.parseASCII=function(e){
var t=/facet([\s\S]*?)endfacet/g,r=[],o=[];
for(;null!==(n=t.exec(e));){
var n,i=n[1],
s=/vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/g;
for(;null!==(l=s.exec(i));){
var l;
r.push(parseFloat(l[1]),parseFloat(l[2]),parseFloat(l[3]))}}
var f=new THREE.BufferGeometry;
return f.setAttribute("position",new THREE.Float32BufferAttribute(r,3)),f.computeVertexNormals(),f};
