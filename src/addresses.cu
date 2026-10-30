// Ed25519 field arithmetic adapted from public-domain TweetNaCl (see
// THIRD_PARTY_NOTICES). Fixed-base multiplication uses a reproducible public
// 4-bit table reproduced by examples/generate_ed25519_table.rs, not a secret cache.
typedef long long i64;
typedef i64 gf[16];
__device__ void fcopy(gf o,const gf a){for(int i=0;i<16;i++)o[i]=a[i];}
__device__ void carry(gf o){
 for(int i=0;i<16;i++){o[i]+=65536;i64 c=o[i]>>16;o[(i+1)*(i<15)]+=c-1+37*(c-1)*(i==15);o[i]-=c*65536;}
}
__device__ void fm(gf o,const gf a,const gf b){
 i64 t[31]={0};
 #pragma unroll
 for(int i=0;i<16;i++) {
   #pragma unroll
   for(int j=0;j<16;j++)t[i+j]+=a[i]*b[j];
 }
 #pragma unroll
 for(int i=0;i<15;i++)t[i]+=38*t[i+16];
 #pragma unroll
 for(int i=0;i<16;i++)o[i]=t[i];
 carry(o);carry(o);
}
__device__ void fs(gf o,const gf a){fm(o,a,a);}
__device__ void square_n(gf o,const gf a,int n){fcopy(o,a);for(int i=0;i<n;i++)fs(o,o);}
__device__ void inverse(gf o,const gf z){
 // Addition chain for z^(2^255-21): 254 squarings and a small number of
 // multiplications, instead of multiplying on nearly every exponent bit.
 gf a,b,c,z11,q10,q50;
 fs(a,z);fs(b,a);fs(b,b);fm(b,b,z);fm(z11,b,a);
 fs(a,z11);fm(a,a,b); // 2^5-1
 square_n(b,a,5);fm(q10,b,a);
 square_n(b,q10,10);fm(a,b,q10); // 2^20-1
 square_n(b,a,20);fm(a,b,a); // 2^40-1
 square_n(b,a,10);fm(q50,b,q10);
 square_n(b,q50,50);fm(a,b,q50); // 2^100-1
 square_n(b,a,100);fm(a,b,a); // 2^200-1
 square_n(b,a,50);fm(c,b,q50); // 2^250-1
 square_n(b,c,5);fm(o,b,z11);
}
__device__ void field_pack(unsigned char *o,const gf n){
 gf t,m;fcopy(t,n);carry(t);carry(t);carry(t);
 for(int j=0;j<2;j++){
   m[0]=t[0]-0xffed;
   for(int i=1;i<15;i++){m[i]=t[i]-0xffff-((m[i-1]>>16)&1);m[i-1]&=0xffff;}
   m[15]=t[15]-0x7fff-((m[14]>>16)&1);i64 bit=(m[15]>>16)&1;m[14]&=0xffff;
   i64 mask=-(1-bit);for(int i=0;i<16;i++)t[i]^=mask&(t[i]^m[i]);
 }
 for(int i=0;i<16;i++){o[i*2]=t[i];o[i*2+1]=t[i]>>8;}
}
__device__ void mixed_add(gf *p,const unsigned short *table,unsigned int entry){
 gf a,b,c,d,t;
 for(int i=0;i<16;i++){a[i]=p[1][i]-p[0][i];b[i]=p[1][i]+p[0][i];t[i]=__ldg(table+entry*48+16+i);}
 fm(a,a,t);
 for(int i=0;i<16;i++)t[i]=__ldg(table+entry*48+i);
 fm(b,b,t);
 for(int i=0;i<16;i++)t[i]=__ldg(table+entry*48+32+i);
 fm(c,p[3],t);
 for(int i=0;i<16;i++){d[i]=2*p[2][i];t[i]=b[i]-a[i];a[i]=b[i]+a[i];b[i]=d[i]-c[i];c[i]=d[i]+c[i];}
 fm(p[0],t,b);fm(p[1],c,a);fm(p[3],t,a);fm(p[2],b,c);
}
__device__ void public_key(const unsigned char *seed,const unsigned short *table,unsigned char *out){
 u64 digest[8];short_hash(seed,32,digest);
 unsigned char scalar[32];for(int j=0;j<4;j++)for(int k=0;k<8;k++)scalar[j*8+k]=digest[j]>>(56-8*k);
 scalar[0]&=248;scalar[31]&=127;scalar[31]|=64;
 gf p[4]={0};p[1][0]=1;p[2][0]=1;
 for(int window=0;window<64;window++){
   unsigned int digit=(scalar[window/2]>>(4*(window&1)))&15;
   mixed_add(p,table,window*16+digit);
 }
 gf zi,x,y;inverse(zi,p[2]);fm(x,p[0],zi);fm(y,p[1],zi);
 unsigned char xx[32];field_pack(xx,x);field_pack(out,y);out[31]^=(xx[0]&1)<<7;
}

__device__ __constant__ unsigned int SHA256_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ __forceinline__ unsigned int r32(unsigned int x,int n){return (x>>n)|(x<<(32-n));}
__device__ void sha256_short(const unsigned char *input,int len,unsigned char *out){
 unsigned int s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 int blocks=(len+9+63)/64;
 for(int block=0;block<blocks;block++){
   unsigned int w[16];
   #pragma unroll
   for(int j=0;j<16;j++){unsigned int v=0;for(int k=0;k<4;k++){int pos=block*64+j*4+k;v=(v<<8)|(pos<len?input[pos]:(pos==len?128:0));}w[j]=v;}
   if(block==blocks-1)w[15]=len*8;
   unsigned int a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
   #pragma unroll
   for(int j=0;j<64;j++){
     if(j>=16){unsigned int x=w[(j+1)&15],y=w[(j+14)&15];w[j&15]+=(r32(x,7)^r32(x,18)^(x>>3))+w[(j+9)&15]+(r32(y,17)^r32(y,19)^(y>>10));}
     unsigned int t1=h+(r32(e,6)^r32(e,11)^r32(e,25))+((e&f)^(~e&g))+SHA256_K[j]+w[j&15];
     unsigned int t2=(r32(a,2)^r32(a,13)^r32(a,22))+((a&b)^(a&c)^(b&c));
     h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
   }
   s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
 }
 for(int j=0;j<8;j++)for(int k=0;k<4;k++)out[j*4+k]=s[j]>>(24-8*k);
}
extern "C" __global__ void wallet_addresses(const unsigned char *seeds,const unsigned short *table,unsigned char *addresses,unsigned int n){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned char pk[32],data[43],state[71],hash[32],raw[36];
 public_key(seeds+i*64,table,pk);
 for(int j=0;j<43;j++)data[j]=TON_DATA[j];
 for(int j=0;j<32;j++){data[10+j]|=pk[j]>>1;data[11+j]|=pk[j]<<7;}
 sha256_short(data,43,hash);
 for(int j=0;j<71;j++)state[j]=TON_STATE[j];
 for(int j=0;j<32;j++)state[39+j]=hash[j];
 sha256_short(state,71,hash);raw[0]=0x51;raw[1]=0;
 for(int j=0;j<32;j++)raw[2+j]=hash[j];
 unsigned short crc=0;
 for(int j=0;j<34;j++){crc^=(unsigned short)raw[j]<<8;for(int k=0;k<8;k++)crc=(crc<<1)^((crc&0x8000)?0x1021:0);}
 raw[34]=crc>>8;raw[35]=crc;
 for(int j=0;j<36;j++)addresses[i*36+j]=raw[j];
}
extern "C" __global__ void filter_addresses(const unsigned char *addresses,unsigned int *winner,unsigned int *checksum,u64 mask,u64 value,int bits,unsigned int n){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const unsigned char *p=addresses+i*36;
 u64 tail=0;for(int j=30;j<36;j++)tail=(tail<<8)|p[j];
 // This public reduction forces the complete address pipeline to execute during
 // benchmarks too, when there is deliberately no suffix/winner to return.
 atomicXor(checksum,(unsigned int)tail);
 if(bits>0 && (tail&mask)==value)atomicMin(winner,i);
}
