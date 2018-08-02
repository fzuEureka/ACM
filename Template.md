# 模板

### 马拉车 最大回文串
```
char s_new[220005],s[110005];
int p[220005];

int manacher()
{
    int len = strlen(s_new);
    int max_len = -1;
    int id;
    int mx = 0;
    for (int i = 1; i < len; i++){
        if (i < mx)
            p[i] = min(p[2 * id - i], mx - i);
        else
            p[i] = 1;
        while (s_new[i - p[i]] == s_new[i + p[i]])
            p[i]++;
        if (mx < i + p[i])
        {
            id = i;
            mx = i + p[i];
        }
        max_len = max(max_len, p[i] - 1);
    }
    return max_len;
}
```
### KMP
```
const int maxn=10000005;
int kmp_next[maxn];
void getNext(string p)
{
    int siz=p.size();
    int i=0,j=-1;
    kmp_next[0]=-1;
    while(i<siz)
    {

        if(j==-1||p[i]==p[j])
        {
            kmp_next[++i]=++j;
        }
        else j=kmp_next[j];
    }
}
int KMP(string T,string p)
{
    getNext(p);
    int T_siz=T.size();
    int i,j=0;
    int m=p.size();
    int ans=0;
    for(i=0;i<T_siz;++i)
    {
        while(j!=-1&&T[i]!=p[j])j=kmp_next[j];
        j++;

        if(j>=m)
        {
            //匹配到了
             j=kmp_next[j];
             ans++;
             cout<<i<<endl;
        }
    }
    return ans;
}
```


### 二分图匹配
```
int 
vis[405],match[405];
vector<int>G[405];
bool dfs(int x){
    vis[x]=1;
    for(int i=0;i<G[x].size();++i){
        int u=match[G[x][i]];
        if(vis[u])continue;
        if(u==-1||dfs(u)){
            match[G[x][i]]=x;
            return true;
        }
    }
    return false;
}
int search(){
    int res=0;
    for(int i=1;i<=n;++i){
            memset(vis,0, sizeof(vis));
            if(dfs(i))res++;
    }
    return  res;
}
```
### 线段树
```
const int maxn=1e5+7;
long long sum[maxn<<2],add[maxn<<2];
long long a[maxn],n,rt=1;
//向上更新区间和
void PushUp(int rt){
    sum[rt]=sum[rt<<1]+sum[rt<<1|1];
}
//树状数组 
void Build(int l,int r,int rt){

    if(l==r){
        sum[rt]=a[l];
        return;
    }
    int m=(l+r)>>1;
    Build(l,m,rt<<1);
    Build(m+1,r,rt<<1|1);
    PushUp(rt);
}
//点修改
void Update(int L,int c,int l,int r,int rt){
    if(l==r){
        sum[rt]+=c;
        return ;
    }
    int m=(l+r)>>1;
    if(L<=m)Update(L,c,l,m,rt<<1);
    else Update(L,c,m+1,r,rt<<1|1);
    PushUp(rt);
}
//下推标记
void PushDown(int rt,int ln,int rn){
    if(add[rt]){
        add[rt<<1]+=add[rt];
        add[rt<<1|1]+=add[rt];
        sum[rt<<1]+=add[rt]*ln;
        sum[rt<<1|1]+=add[rt]*rn;
        add[rt]=0;
    }
}
//区间修改
void Update(int L,int R,int c,int l,int r,int rt){
    if(L<=l&&R>=r){
        sum[rt]+=(r-l+1)*c;
        add[rt]+=c;
        return ;
    }
    int m=(l+r)>>1;
    PushDown(rt,m-l+1,r-m);
    if(L<=m)Update(L,R,c,l,m,rt<<1);
    if(R>m)Update(L,R,c,m+1,r,rt<<1|1);
    PushUp(rt);
}
//区间查询
long long Query(int L,int R,int l,int r,int rt){
    if(L<=l&&r<=R){
        return sum[rt];
    }
    int m=(l+r)>>1;
    long long  ans=0;
    PushDown(rt,m-l+1,r-m); //记得下推标记
    if(L<=m)ans+=Query(L,R,l,m,rt<<1);
    if(R>m)ans+=Query(L,R,m+1,r,rt<<1|1);
    return ans;
}
```

### Tarjan 缩点
```
int n,m,top,tot,sum,stk[maxn],color[maxn],cnt[maxn],dfn[maxn],low[maxn],vis[maxn],du[maxn];
void tarjan(int v){
    vis[v]=1;
    stk[++top]=v;
    dfn[v]=low[v]=++tot;
    for(int i=0;i<G[v].size();++i){
        if(!dfn[G[v][i]]){  
            tarjan(G[v][i]);
            low[v]=min(low[v],low[G[v][i]]);
        }else if(vis[G[v][i]]){    
            low[v]=min(low[v],low[G[v][i]]);
        }
    }
    if(dfn[v]==low[v]){
        color[v]=++sum; //缩点 染色 sum记录强联通分量
        vis[v]=0;
        while(stk[top]!=v){
            vis[stk[top]]=0;
            color[stk[top]]=sum;
            top--;
        }
        top--;
    }
}
```

### 凸包 graham_scan
```
const int maxn = 1e5 + 5;
const double PI = acos(-1.0);
struct point {
	double x;
	double y;
}p[maxn];
vector<point>G;
//叉积 判断左转右转 左转为正
double cross_product(point p0, point p1, point p2) {
	return (p1.x - p0.x)*(p2.y - p0.y) - (p2.x - p0.x)*(p1.y - p0.y);
}
double dis(point a, point b) {
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
bool cmp(point a, point b) {
	double tmp = cross_product(p[0], a, b); //按极角大小排序，极角相同按距离小的先排;
	if (fabs(tmp) < 1e-7) {
		return dis(p[0], a) < dis(p[0], b);
	}
	return tmp > 0;
}
void graham_scan(int n) {
	int top = 2;
	int index = 0;
	for (int i = 1; i < n; ++i)
	{
        //找出p[0]，纵坐标最小，当纵坐标一样，横坐标最小
		if (p[i].y < p[index].y || (p[i].y == p[index].y && p[i].x < p[index].x))
		{
			index = i;
		}
	}
	swap(p[0], p[index]);
	G.push_back(p[0]);
	sort(p + 1, p + n, cmp);
	G.push_back(p[1]);
	G.push_back(p[2]);
	for (int i = 3; i < n; ++i)
	{
        //当发生非左转将栈顶退出 要判断栈中个数，共线时候可能发生，只剩下一个点在栈中
		while (top > 0 && cross_product(G[top - 1], p[i], G[top]) >= 0) 
		{
			--top;
			G.pop_back();
		}
		G.push_back(p[i]);
		++top;
	}
}
//计算凸包周长
double cal() {
	double res = 0.0;
	G.push_back(p[0]);
	for (int i = 1; i < G.size(); ++i) {
		res += dis(G[i], G[i - 1]);
	}
	return res;
}
```