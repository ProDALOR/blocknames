// PBKDF2-HMAC-SHA512, one independent 64-byte password per CUDA thread.
// State uses structure-of-arrays layout for coalesced memory accesses.
typedef unsigned long long u64;
__device__ __constant__ u64 K[80] = {
0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL};
__device__ __forceinline__ u64 ror(u64 x, int n) { return (x >> n) | (x << (64-n)); }
__device__ __forceinline__ void iv(u64 *s) {
 s[0]=0x6a09e667f3bcc908ULL; s[1]=0xbb67ae8584caa73bULL;
 s[2]=0x3c6ef372fe94f82bULL; s[3]=0xa54ff53a5f1d36f1ULL;
 s[4]=0x510e527fade682d1ULL; s[5]=0x9b05688c2b3e6c1fULL;
 s[6]=0x1f83d9abfb41bd6bULL; s[7]=0x5be0cd19137e2179ULL;
}
__device__ __forceinline__ void compress(u64 *s, u64 *w) {
 u64 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
 #pragma unroll
 for(int i=0;i<80;i++) {
   if(i>=16) {
     u64 x=w[(i+1)&15],y=w[(i+14)&15];
     w[i&15] += (ror(x,1)^ror(x,8)^(x>>7)) + w[(i+9)&15] + (ror(y,19)^ror(y,61)^(y>>6));
   }
   u64 t1=h+(ror(e,14)^ror(e,18)^ror(e,41))+((e&f)^(~e&g))+K[i]+w[i&15];
   u64 t2=(ror(a,28)^ror(a,34)^ror(a,39))+((a&b)^(a&c)^(b&c));
   h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
 }
 s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}
__device__ __forceinline__ void hash64(const u64 *prefix, const u64 *msg, u64 *out) {
 u64 w[16];
 #pragma unroll
 for(int j=0;j<8;j++) { out[j]=prefix[j];w[j]=msg[j]; }
 w[8]=0x8000000000000000ULL;
 #pragma unroll
 for(int j=9;j<15;j++)w[j]=0;
 w[15]=1536; // 128-byte HMAC prefix + 64-byte message
 compress(out,w);
}
// ipad/opad SHA-512 states depend on the candidate key, but not on the salt.
// Compute them once per candidate, then reuse them for every PBKDF2 iteration.
__device__ __forceinline__ void prefixes(const unsigned char *keys, unsigned int i, u64 *in, u64 *out) {
 u64 w[16];
 #pragma unroll
 for(int j=0;j<16;j++) {
   u64 key=0;
   if(j<8) for(int k=0;k<8;k++)key=(key<<8)|keys[i*64+j*8+k];
   w[j]=key^0x3636363636363636ULL;
 }
 iv(in);compress(in,w);
 #pragma unroll
 for(int j=0;j<16;j++) {
   u64 key=0;
   if(j<8) for(int k=0;k<8;k++)key=(key<<8)|keys[i*64+j*8+k];
   w[j]=key^0x5c5c5c5c5c5c5c5cULL;
 }
 iv(out);compress(out,w);
}
__device__ __forceinline__ void first(const u64 *in, const u64 *out, const unsigned char *salt, int saltlen, u64 *u) {
 u64 w[16];
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=0;
 for(int j=0;j<saltlen;j++)w[j/8]|=(u64)salt[j]<<(56-8*(j%8));
 int end=saltlen+4;
 w[(end-1)/8]|=(u64)1<<(56-8*((end-1)%8)); // block number 1
 w[end/8]|=(u64)0x80<<(56-8*(end%8));
 w[15]=(128+end)*8;
 u64 tmp[8];
 #pragma unroll
 for(int j=0;j<8;j++)tmp[j]=in[j];
 compress(tmp,w);hash64(out,tmp,u);
}
__device__ __forceinline__ void prepare(const unsigned char *keys, const unsigned char *salt, int saltlen, unsigned int i, u64 *in, u64 *out, u64 *u) {
 prefixes(keys,i,in,out);first(in,out,salt,saltlen,u);
}
// Both TON salts are exactly 16 ASCII bytes. Cache their complete padded U1
// message as immediates: salt || BE32(1) || SHA padding, following 128 HMAC bytes.
__device__ __forceinline__ void first_fixed(const u64 *in, const u64 *out, bool wallet, u64 *u) {
 u64 w[16]={0},tmp[8];
 w[0]=wallet?0x544f4e2064656661ULL:0x544f4e2073656564ULL;
 w[1]=wallet?0x756c742073656564ULL:0x2076657273696f6eULL;
 w[2]=0x0000000180000000ULL;w[15]=1184;
 #pragma unroll
 for(int j=0;j<8;j++)tmp[j]=in[j];
 compress(tmp,w);hash64(out,tmp,u);
}
extern "C" __global__ void init(const unsigned char *keys, const unsigned char *salt, int saltlen, u64 *state, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 u64 in[8],out[8],u[8];
 prepare(keys,salt,saltlen,i,in,out,u);
 #pragma unroll
 for(int j=0;j<8;j++) {state[j*n+i]=in[j];state[(8+j)*n+i]=out[j];state[(16+j)*n+i]=u[j];state[(24+j)*n+i]=u[j];}
}

// Initialize derivation directly from passing raw-candidate indexes in VRAM.
// This replaces downloading indexes, gathering entropy on CPU, and reuploading it.
extern "C" __global__ void init_indexed(const unsigned char *keys, const unsigned int *indexes, u64 *state, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 u64 in[8],out[8],u[8];
 prefixes(keys,indexes[i],in,out);first_fixed(in,out,true,u);
 #pragma unroll
 for(int j=0;j<8;j++){state[j*n+i]=in[j];state[(8+j)*n+i]=out[j];state[(16+j)*n+i]=u[j];state[(24+j)*n+i]=u[j];}
}

// SHA-512 for a short variable-length byte string. TON mnemonic text is at most
// 215 bytes (24 eight-letter words + spaces); at most two SHA blocks are needed.
__device__ __forceinline__ void short_hash(const unsigned char *text, int len, u64 *out) {
 iv(out);
 int blocks=(len+17+127)/128;
 for(int b=0;b<blocks;b++) {
   u64 w[16];
   #pragma unroll
   for(int j=0;j<16;j++) {
     u64 v=0;
     #pragma unroll
     for(int k=0;k<8;k++) {
       int pos=b*128+j*8+k;
       unsigned char c=pos<len?text[pos]:(pos==len?0x80:0);
       v=(v<<8)|c;
     }
     w[j]=v;
   }
   if(b==blocks-1)w[15]=(u64)len*8;
   compress(out,w);
 }
}

// Dictionary entries are [length, eight padded ASCII bytes]. Random word lookups
// use read-only global memory rather than divergent constant-memory broadcasts.
// Input words remain reproducible from OS-generated indexes kept by the host.
extern "C" __global__ void phrase_entropy(const unsigned short *indexes, const unsigned char *dictionary, unsigned char *entropy, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned char text[256];int len=0;
 for(int j=0;j<24;j++) {
   unsigned int index=indexes[i*24+j];
   if(j)text[len++]=' ';
   int size=__ldg(dictionary+index*9);
   for(int k=0;k<size;k++)text[len++]=__ldg(dictionary+index*9+1+k);
 }
 // TON entropy is HMAC-SHA512(key=mnemonic text, message=empty), not BIP39 KDF.
 // HMAC hashes keys longer than its 128-byte block before applying ipad/opad.
 u64 key[16],w[16],in[8],out[8],u[8],tmp[8];
 #pragma unroll
 for(int j=0;j<16;j++)key[j]=0;
 if(len>128)short_hash(text,len,key);
 else for(int j=0;j<len;j++)key[j/8]|=(u64)text[j]<<(56-8*(j%8));
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=key[j]^0x3636363636363636ULL;
 iv(in);compress(in,w);
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=key[j]^0x5c5c5c5c5c5c5c5cULL;
 iv(out);compress(out,w);
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=0;
 w[0]=0x8000000000000000ULL;w[15]=1024;
 #pragma unroll
 for(int j=0;j<8;j++)tmp[j]=in[j];
 compress(tmp,w);hash64(out,tmp,u);
 #pragma unroll
 for(int j=0;j<8;j++)for(int k=0;k<8;k++)entropy[i*64+j*8+k]=u[j]>>(56-8*k);
}
extern "C" __global__ void step(u64 *state, unsigned int n, unsigned int rounds) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 u64 in[8],out[8],u[8],t[8],tmp[8];
 #pragma unroll
 for(int j=0;j<8;j++){in[j]=state[j*n+i];out[j]=state[(8+j)*n+i];u[j]=state[(16+j)*n+i];t[j]=state[(24+j)*n+i];}
 for(unsigned int r=0;r<rounds;r++) {
   hash64(in,u,tmp);hash64(out,tmp,u);
   #pragma unroll
   for(int j=0;j<8;j++)t[j]^=u[j];
 }
 #pragma unroll
 for(int j=0;j<8;j++){state[(16+j)*n+i]=u[j];state[(24+j)*n+i]=t[j];}
}
extern "C" __global__ void finish(const u64 *state, unsigned char *result, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int j=0;j<8;j++)for(int k=0;k<8;k++)result[i*64+j*8+k]=state[(24+j)*n+i]>>(56-8*k);
}
// Validation only needs the first byte of the 390-round derived key. Keep HMAC
// state in registers and return compact passing indexes, not 64 bytes per phrase.
extern "C" __global__ void validate(const unsigned char *keys, unsigned int *indexes, unsigned int *count, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 u64 in[8],out[8],u[8],tmp[8];
 prefixes(keys,i,in,out);first_fixed(in,out,false,u);
 u64 first=u[0];
 for(int r=1;r<390;r++) {
   hash64(in,u,tmp);hash64(out,tmp,u);first^=u[0];
 }
 if((first>>56)==0)indexes[atomicAdd(count,1u)]=i;
}

// HMAC setup for short keys. Twelve English BIP39 words occupy <=107 bytes,
// below SHA-512's 128-byte HMAC block. No SHA512(key) preprocessing is needed.
__device__ __forceinline__ void text_prefixes(const unsigned char *key, int len, u64 *in, u64 *out) {
 u64 words[16]={0},w[16];
 for(int j=0;j<len;j++)words[j/8]|=(u64)key[j]<<(56-8*(j%8));
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=words[j]^0x3636363636363636ULL;
 iv(in);compress(in,w);
 #pragma unroll
 for(int j=0;j<16;j++)w[j]=words[j]^0x5c5c5c5c5c5c5c5cULL;
 iv(out);compress(out,w);
}
__device__ __forceinline__ void hmac_short(const unsigned char *key, int keylen, const unsigned char *msg, int len, u64 *result) {
 u64 in[8],out[8],w[16]={0},tmp[8];
 text_prefixes(key,keylen,in,out);
 for(int j=0;j<len;j++)w[j/8]|=(u64)msg[j]<<(56-8*(j%8));
 w[len/8]|=(u64)0x80<<(56-8*(len%8));w[15]=(128+len)*8;
 #pragma unroll
 for(int j=0;j<8;j++)tmp[j]=in[j];
 compress(tmp,w);hash64(out,tmp,result);
}
extern "C" __global__ void bip39_init(const unsigned short *indexes, const unsigned char *dictionary, u64 *state, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned char text[128];int len=0;
 for(int j=0;j<12;j++) {
   unsigned int index=indexes[i*24+j];if(j)text[len++]=' ';
   int size=__ldg(dictionary+index*9);
   for(int k=0;k<size;k++)text[len++]=__ldg(dictionary+index*9+1+k);
 }
 u64 in[8],out[8],u[8],w[16]={0},tmp[8];
 text_prefixes(text,len,in,out);
 // Empty BIP39 passphrase: salt="mnemonic", followed by block number 1.
 w[0]=0x6d6e656d6f6e6963ULL;w[1]=0x0000000180000000ULL;w[15]=1120;
 #pragma unroll
 for(int j=0;j<8;j++)tmp[j]=in[j];
 compress(tmp,w);hash64(out,tmp,u);
 #pragma unroll
 for(int j=0;j<8;j++){state[j*n+i]=in[j];state[(8+j)*n+i]=out[j];state[(16+j)*n+i]=u[j];state[(24+j)*n+i]=u[j];}
}
extern "C" __global__ void bip39_slip10(const u64 *state, unsigned char *result, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const unsigned char label[12]={'e','d','2','5','5','1','9',' ','s','e','e','d'};
 u64 in[8],out[8],seed[8],tmp[8],node[8];
 text_prefixes(label,12,in,out);
 #pragma unroll
 for(int j=0;j<8;j++)seed[j]=state[(24+j)*n+i];
 hash64(in,seed,tmp);hash64(out,tmp,node);
 const unsigned int path[3]={0x8000002cU,0x8000025fU,0x80000000U};
 for(int level=0;level<3;level++) {
   unsigned char chain[32],data[37];data[0]=0;
   #pragma unroll
   for(int j=0;j<4;j++)for(int k=0;k<8;k++) {
     data[1+j*8+k]=node[j]>>(56-8*k);chain[j*8+k]=node[4+j]>>(56-8*k);
   }
   for(int j=0;j<4;j++)data[33+j]=path[level]>>(24-8*j);
   hmac_short(chain,32,data,37,node);
 }
 #pragma unroll
 for(int j=0;j<8;j++)for(int k=0;k<8;k++)result[i*64+j*8+k]=j<4?(node[j]>>(56-8*k)):0;
}
