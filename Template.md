# 模板

### 马拉车 最大回文串
```c++
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
```c++
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
- 最小点覆盖==最大匹配数 |最小点集|=|最大匹配|
- 最小路径覆盖==总点集合G-最大匹配数
-  |最大独立集| = |V|-|最大匹配数|
```c++
const int maxn=1e5;
int vis[maxn],match[maxn];
vector<int>G[maxn];
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
    memset(match,-1,sizeof(match));
    for(int i=1;i<=n;++i){
            memset(vis,0, sizeof(vis));
            if(dfs(i))res++;
    }
    return  res;
}
```

### 二位树状数组求动态二维矩阵和
```c++
//#define lowbit(a) (a&(-a))
ll c[N][N];
void update(int x,int y,int z){
	for(int i=x;i<=n;i+=lowbit(i))
		for(int j=y;j<=m;j+=lowbit(j))
			c[i][j]+=z;
}
ll sum(int x,int y){
	ll ret=0;
	for(int i=x;i>=1;i-=lowbit(i))
		for(int j=y;j>=1;j-=lowbit(j))
			ret+=c[i][j];
	return ret;
}
```


### 线段树
```c++
const int maxn=1e5+7;
long long sum[maxn<<2],add[maxn<<2];
long long a[maxn],n,rt=1;
//向上更新区间和
void PushUp(int rt){
    sum[rt]=sum[rt<<1]+sum[rt<<1|1];
}

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
### 扫描线+线段树求矩形并面积
```c++
struct segment{
	vector<ll> G; //x坐标的值，用来离散
	ll sta[N],sum[N]; //sum是区间有效长度，sta是区间状态，不为0说明全被覆盖
	void pushup(int l,int r,int rt){
		if(sta[rt]){
			sum[rt]=G[r]-G[l-1]; //求出原坐标中的长度
		}	
		else  if(l==r)sum[rt]=0;
		else sum[rt]=sum[rt<<1]+sum[rt<<1|1];
	} 
	void update(int ql,int qr,int f,int l,int r,int rt){
		if(ql<=l&&r<=qr){
			sta[rt]+=f;
			pushup(l,r,rt);
			return ;
		}
		int mid=(l+r)/2;
		if(ql<=mid)update(ql,qr,f,l,mid,rt<<1);
		if(qr>mid)update(ql,qr,f,mid+1,r,rt<<1|1);
		pushup(l,r,rt);	
	}
}tre;
```
### Tarjan 缩点
```c++
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
// 无向图边双
struct tarjan{
	int tot=0,cnt=0,tim=0,stk[N],dfn[N],low[N];
	void Tarjan_EBC(int u,int fa){
		vis[u]=true;
		stk[++tot]=u;
		dfn[u]=low[u]=++tim;
		for(auto v:G[u]){
			if(!dfn[v]){
				Tarjan_EBC(v,u);
				low[u]=min(low[u],low[v]);
			}
			else if(v!=fa&&vis[v])
				low[u]=min(low[u],dfn[v]);
		}
		if(dfn[u]==low[u]){
			col[u]=++cnt;
			while(stk[tot]!=u){
				col[stk[tot]]=cnt;
				vis[stk[tot]]=false;
				tot--;
			}
			tot--;
		}
	}
};
```

### 凸包 graham_scan
```c++
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

### st表
- 区间最大最小值 stmax[i][j] 表示i~i+2^j-1 的最大值
```c++
const int maxn = 2e5 + 5;
int a[maxn], b[maxn], n, stmax[maxn][20], stmin[maxn][20];
void ST() {
	for (int i = 1; i <= n; i++)
	{
		stmax[i][0] = a[i];
		stmin[i][0] = b[i];
	}
	for (int j = 1; (1 << j) <= n; j++)
	{
		for (int i = 1; i + (1 << j) - 1 <= n; i++)
		{
			stmax[i][j] = max(stmax[i][j - 1], stmax[i + (1 << (j - 1))][j - 1]);
			stmin[i][j] = min(stmin[i][j - 1], stmin[i + (1 << (j - 1))][j - 1]);
		}
	}
}
int queryMin(int s, int e) {
	int nlog = 0;
	while ((1 << (nlog + 1)) <= e - s + 1)nlog++;
    //int nlog = int(log(e - s + 1) / log(2));
	return min(stmin[s][nlog], stmin[e - (1 << nlog) + 1][nlog]);
}
int queryMax(int s, int e) {
	int nlog = 0;
	while ((1 << (nlog + 1)) <= e - s + 1)nlog++;
    //int nlog = int(log(e - s + 1) / log(2));
	return  max(stmax[s][nlog], stmax[e - (1 << nlog) + 1][nlog]);
}
```
### 离线求lca（最近公共祖先） tarjan
- 对每个节点 利用并查集建一颗子树，搜索道根节点向上回溯，查到第二点的时候find向上查父节点就是他们最近的子树根节点，就是答案

```c++
//op和G是pair对存边和要求的q对lca,op第二维是第i对要求lca的点对,ans[i]第i对的lca
struct LCA{
	vector<pii>G[N],op[N];
	int pre[N],ans[N],dis[N];
	bool vis[N];
	int find_set(int x){
		if(pre[x]==x)return x;
		return pre[x]=find_set(pre[x]);
	}
	void tarjan(int u){
		vis[u]=true;
		for(auto it:G[u]){
			if(!vis[it.fi]){
				dis[it.fi]=dis[u]+it.se;
				tarjan(it.fi);
				pre[it.fi]=u;
			} 
		}
		for(auto it:op[u]){
			if(vis[it.fi]){
				ans[it.se]=find_set(it.fi);
			}
		}
	}
};
```
```
### 在线LCA
- [lca详解1](https://blog.csdn.net/y990041769/article/details/40887469)
- [lca详解2](http://www.cnblogs.com/scau20110726/archive/2013/05/26/3100812.html)
```c++
const int N=1e5+5;
int dp[N][20],n;  //dp[i][j] 下标i开始  长度为2^j区间中最小值的下标 
vector<int> G[N];
int tot,ver[N<<1],R[N<<1],First[N<<1];
//ver:节点编号 R：ver中的点下标对应的深度 first：第一次搜到该点在ver数组中的下标 
bool vis[N]; 
void dfs(int u,int dep){

	vis[u]=true;ver[++tot]=u;First[u]=tot;R[tot]=dep;
	for(int i=0;i<G[u].size();++i){
		int v=G[u][i];
		if(!vis[v]){
			dfs(v,dep+1);
			ver[++tot]=u;
			R[tot]=dep;
		}
	}
	return;
}
void ST(){
	for(int i=1;i<=tot;++i){
		dp[i][0]=i;
	}
	for(int j=1;(1<<j)<=tot;++j){
		for(int i=1;i+(1<<j)-1<=tot;i++){
			int a=dp[i][j-1];
			int b=dp[i+(1<<(j-1))][j-1];
			if(R[a]<R[b])dp[i][j]=a;
			else dp[i][j]=b;
		}
		 
	} 
}
int RMQ(int l,int r){
	int k=0;
	while((1<<(k+1))<=r-l+1)
		k++;
	int a=dp[l][k],b=dp[r-(1<<k)+1][k];
	return R[a]<R[b]?a:b;
}
int LCA(int u,int v){
	int x=First[u],y=First[v];
	if(x>y)swap(x,y);
	int res=RMQ(x,y);
	return ver[res];
}
```

### 最小生成树
```c++
struct node{
	ll cost;
	int u,v;
}p[maxn];
void kruskal(){
	int k=0;
	for(int i=1;i<=tot;++i){
		int u=p[i].u;
		int v=p[i].v;
		int fx=find_set(u),fy=find_set(v);
		if(fx!=fy){
			pre[fx]=fy;
			G[u].push_back(v);
			G[v].push_back(u);
			k++;
			if(k==n*m-1)break; 
		}
		 
	}
}
```
### 扩展kmp 
- 定义母串S，和字串T，设S的长度为n，T的长度为m，求T与S的每一个后缀的最长公共前缀，也就是说，设extend数组,extend[i]表示T与S[i,n-1]的最长公共前缀，要求出所有extend[i](0<=i<n)。
- 算法详解 [扩展kmp总结](https://blog.csdn.net/dyx404514/article/details/41831947)

```c++
typedef long long ll;
const int maxn = 1e6+5;
string S, T;
ll nxt[maxn], ex[maxn];

void getNext() {
	int siz = T.size();
	nxt[0] = siz;
	int i = 0;
	while (T[i] == T[i + 1] && i + 1 < siz)i++;
	nxt[1] = i;
	int p0 = 1;
	for (int i = 2; i < siz; ++i) {

		if (nxt[i - p0] + i < nxt[p0] + p0)
			nxt[i] = nxt[i - p0];
		else {
			int j = nxt[p0] + p0 - i;
			if (j < 0)j = 0;
			while (i + j < siz&&T[j] == T[j + i])j++;
			nxt[i] = j;
			p0= i;
		}
	}
}
void EXKMP() {
	int size1 = S.size(), i = 0, size2 = T.size();
	getNext();
	while (S[i] == T[i] && i < size1&&i < size2)i++;
	ex[0] = i;
	int p0 = 0;
	for (int i = 1; i < size1; ++i) {
		if (i + nxt[i - p0] < ex[p0] + p0)
			ex[i] = nxt[i - p0];
		else {
			int j = ex[p0] + p0 - i;
			if (j < 0)j = 0;
			while (i + j < size1&&j < size2&&T[j] == S[i + j])
				j++;
			ex[i] = j;
			p0 = i;
		}
	}
}
```

### Treap
-  [treap详解](https://blog.csdn.net/u014634338/article/details/49612159)

```c++
typedef  struct TreapNode* Tree;
typedef long long ll;
struct TreapNode {
	int val;
	int priority;
	Tree lchild;
	Tree rchild;
	int lsize;
	int rsize;
	TreapNode(int val = 0, int priority = 0) {
		lchild = rchild = NULL;
		lsize = rsize = 0;
		this->val = val;
		this->priority = priority;
	}
};
void left_rotate(Tree &node) {
	Tree temp = node->rchild;
	node->rchild = temp->lchild;
	node->rsize = temp->lsize;
	temp->lsize = node->lsize + node->rsize + 1;
	temp->lchild = node;
	node = temp;
}
void right_rotate(Tree &node) {
	Tree temp = node->lchild;
	node->lchild = temp->rchild;
	node->lsize = temp->rsize;
	temp->rsize = node->lsize + node->rsize + 1;
	temp->rchild = node;
	node = temp;
}
bool insert_val(Tree &root, Tree &node) {
	if (root == NULL) {
		root = node;
		return true;
	}
	else if (root->val <node->val) {     //相同的也要插入改成<=就行
		insert_val(root->rchild, node);
		root->rsize += 1;
		if (root->priority>node->priority)
			left_rotate(root);
		return  true;
	}
	else if (root->val>node->val) {
		insert_val(root->lchild, node);
		root->lsize += 1;
		if (root->priority>node->priority)
			right_rotate(root);
		return true;
	}
}
bool insert(Tree &root, int val, int priority) {
	Tree node = new TreapNode(val, priority);
	return insert_val(root, node);
}
//删除
bool remove(Tree &root, int val)
{

	if (root->val>val) {
		root->lsize -= 1;
		return remove(root->lchild, val);
	}
	else if (root->val<val) {
		root->rsize -= 1;
		return remove(root->rchild, val);
	}
	else
	{

		if (root->lchild == NULL) {
			root = root->rchild;
			return true;
		}
		else if (root->rchild == NULL) {
			root = root->lchild;

			return true;
		}
		if (root->lchild->priority<root->rchild->priority) {
			right_rotate(root); root->rsize -= 1;
			remove(root->rchild, val);

		}
		else {
			left_rotate(root); root->lsize -= 1;
			remove(root->lchild, val);

		}
	}
}
//第k大
int Kth(Tree root, int val) {
	if (root == NULL || val <= 0 || val > root->lsize + root->rsize + 1)return 0;
	if (root->lsize == val - 1)
		return root->val;
	else if (root->lsize>val - 1)return Kth(root->lchild, val);
	else return Kth(root->rchild, val - (root->lsize + 1));
}

//查找是否存在
Tree search(Tree &root, int val)
{
	if (!root)
		return NULL;
	else if (root->val>val)
		return search(root->lchild, val);
	else if (root->val<val)
		return  search(root->rchild, val);
	return root;
}
```

### AC自动机
- [算法详解](https://blog.csdn.net/liu940204/article/details/51347064)
```c++
// HDU 2222 模板题 HDU3065
#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<string>
#include<cstring>
#include<stdio.h>
#include<queue>
#include<algorithm>
using namespace std;
const int maxn=1e6 + 5;
struct node {
    int cnt;
    node *fail;
    node *nxt[26];
    node() {
        cnt = 0;
        fail = NULL;
        for (int i = 0; i < 26; ++i)
            nxt[i] = NULL;
    }
}*que[maxn];
node *root;
char s[maxn];
char t[105];
void Build_trie(char *s) {
    node *p, *q;
    int len = strlen(s);
    p = root;
    for (int i = 0; i < len; ++i) {
        int x = s[i]-'a';
        if (p->nxt[x] == NULL) {
            q = new node;
            p->nxt[x]= q;
        }
        p = p->nxt[x];
    }
    p->cnt++;
}
void Build_AC(node *root) {
    int head=0, tail = 0;
    que[head++] = root;
    while (head != tail) {
        node *p = NULL;
        node *q = que[tail++];
        for (int i = 0; i < 26; ++i) {
            if (q->nxt[i]) {
                if (q == root)q->nxt[i]->fail = root;
                else {
                    p = q->fail;
                    while (p) {
                        if (p->nxt[i] != NULL) {
                            q->nxt[i]->fail = p->nxt[i];
                            break;
                        }
                        p = p->fail;
                    }
                    if (p == NULL)
                        q->nxt[i]->fail = root;
                }
                que[head++] = q->nxt[i];
            }
        }
    }
}
int Query(node *root) {
    int ans = 0;
    int len = strlen(s);
    node *p = root;
    for (int i = 0; i < len; ++i) {
        int x = s[i] - 'a';
        while (p!=root&&p->nxt[x] == NULL)p = p->fail;
        p = p->nxt[x];
        if (p == NULL)p = root;
        node *q = p;
        while (q != root) {
            if (q->cnt >= 0) {
                ans += q->cnt;
                q->cnt = -1;
            }
            else break;
            q = q->fail;
        }
    }
    return ans;
}
int main()
{
    int k;
    scanf("%d", &k);
    while (k--) {
        root = NULL;
        root = new node;

        int n;
        scanf("%d", &n);
        for (int i = 0; i < n; ++i) {
            scanf("%s", t);
            Build_trie(t);
        }
        Build_AC(root);
        scanf("%s", s);
        int ans = Query(root);
        printf("%d\n", ans);
    }
    return 0;
}
```

### 康托展开
- [康托详解](https://zybuluo.com/Junlier/note/1174122)
```c++
int n;  //n个数字 
ll fac[21]; //阶乘 
int b[21];  //排列 
//数字转排列 
void getKT(ll x){
	ll res;
	bool vis[21];
	memset(vis,0,sizeof(vis));
	for(int i=n;i>=1;--i){
		res=x/fac[i-1]+1;   //计算该数是没用过的数字中第几大 
		x=x%fac[i-1];
		rep(j,1,n+1){
			if(!vis[j])res--;
			if(res==0){
				printf("%d ",j);
				vis[j]=true;
				break;
			}
		}
	}
	printf("\n");
}
//排列转数字 
ll invKT(){

	ll res=0;
	rep(i,1,n+1){
		ll cnt=0; 
		rep(j,i+1,n+1)
			if(b[i]>b[j])cnt++;  //有多少没用过的数字比她小 
		res+=cnt*(fac[n-i]);
	}
	res++; //加上本身 
	return res;
}
```
### AVL
- [avl详解](https://www.cnblogs.com/zhuwbox/p/3636783.html)
```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<ll,ll>pii;
typedef vector<int>vi;

#define rep(i,a,b) for(int i=(a);i<(b);i++)
#define fi first
#define se second
#define de(x) cout<<#x<<"="<<x<<endl
#define per(i,a,b) for(int i=(b)-1;i>=(a);--i)
const int N=1e5+5;
struct AvlNode{
	int val;
	AvlNode *lson;
	AvlNode *rson;
	int height;
};
typedef AvlNode* AvlTree;
int Height( AvlTree T)
{
     if(T==NULL)
         return -1;
    return T->height;
}
//传引用才能修改 
void RotateLeft(AvlTree &T){
	AvlTree rt=T->rson;
	T->rson=rt->lson;
	rt->lson=T;
 	T->height = max( Height(T->lson),Height(T->rson) ) + 1;
    rt->height = max( Height(rt->lson),Height(rt->rson) ) + 1;
	T=rt;
}
void RotateRight(AvlTree &T){
	AvlTree rt=T->lson;
	T->lson=rt->rson;
	rt->rson=T;
	T->height = max( Height(T->lson),Height(T->rson) ) + 1;
    rt->height = max( Height(rt->lson),Height(rt->rson) ) + 1;
	T=rt;
}
void RotateLR(AvlTree &T){
	RotateLeft(T->lson);
	RotateRight(T);
} 
void RotateRL(AvlTree &T){
	RotateRight(T->rson);
	RotateLeft(T);
} 
AvlTree ins(AvlTree &T,int x){
	if(T==NULL){
		T=new AvlNode();
		T->val=x;
		T->lson=T->rson=NULL;
		T->height=0;
	}
	else if(x<T->val){
		T->lson=ins(T->lson,x);
		if(Height(T->lson)-Height(T->rson)==2){
			int val=T->lson->val;
			if(x<val){
				RotateRight(T);
			}
			else RotateLR(T);
		}
	} 
	else if(x>T->val){
		T->rson=ins(T->rson,x);
		if(Height(T->rson)-Height(T->lson)==2){
			int val=T->rson->val;
			if(x>val){
				RotateLeft(T);
				
			}
			else RotateRL(T);
		}	
	}
	T->height = max( Height(T->lson),Height(T->rson) ) + 1;
   
	return T;
}
//查找根到节点的路径
void find_val(AvlTree T,int x){


	if(T->val==x)return;
	printf("%d ",T->val);
	if(x<T->val)find_val(T->lson,x);
	else find_val(T->rson,x);
}
int main()
{
	int n;
	AvlTree root=NULL;
	scanf("%d",&n);
	int op,x;
	while(n--){
		scanf("%d%d",&op,&x);
		if(op==1){
			root=ins(root,x);
		}	
		else {
			find_val(root,x);
			printf("%d\n",x);
		}
	}
}
```
### 莫比乌斯
```c++
void mobi(){
	mu[1]=1;
	memset(vis,0,sizeof(vis));
	for(int i=2;i<N;++i){
		if(!vis[i]){
			mu[i]=-1;
			vis[i]=true;
			prime[tot++]=i;
		}
		for(int j=0;j<tot&&prime[j]*i<N;++j){
			vis[i*prime[j]]=1;
			if(i%prime[j]==0)break;
			mu[i*prime[j]]=-mu[i];
		}
	}
}
```
### 树上倍增在线LCA
```c++
void dfs(int x){
	dep[x]=dep[fa[x][0]]+1;
	for(int i=1;i<=mx;++i)
		if(fa[x][i-1])
			fa[x][i]=fa[fa[x][i-1]][i-1];
	for(int i=0;i<G[x].size();++i){
		int v=G[x][i].fi;
		if(v!=fa[x][0]){
			d[v]=d[x]+G[x][i].se;
			fa[v][0]=x;
			dfs(v);
		}
	}
	
}
int LCA(int u,int v){
	if(dep[u]<dep[v])swap(u,v);
	int dif=dep[u]-dep[v];
	for(int i=0;i<=mx;++i)
		if((1<<i)&dif)
			u=fa[u][i];
	if(u==v)return u;
	for(int i=mx;i>=0;--i){
		if(fa[u][i]!=fa[v][i]){
			u=fa[u][i];
			v=fa[v][i];
		}
	}
	return fa[u][0];
}
```

### 高斯消元（残缺版，只写了一定有解的情况,待补充）

```c++
const double eps=-1e9;
double a[105][105]; //系数
void gauss(int n,int m){
	//n个方程组，m列  m-1个未知数 
	int row,col,mxr;
	for(row=col=1;row<=n&&col<=m;++row,++col){
		mxr=row;
		for(int j=row+1;j<=n;++j){
			if(fabs(a[j][col])>fabs(a[mxr][col]))
				mxr=j;
		}
		swap(a[mxr],a[row]);
		if(fabs(a[row][col])<eps){
			row--;
			continue;
		}
		for(int i=row+1;i<=n;++i){  //消成上上角矩阵，要从m->col，因为col的系数是倍数的分母 
			if(fabs(a[i][col])<eps)continue;
			for(int j=m;j>=col;--j)
				a[i][j]-=a[row][j]/a[row][col]*a[i][col];
		}
	}	
	for(int i=n;i>=1;--i){
		for(int j=i+1;j<m;++j){
			a[i][m]-=a[j][m]*a[i][j];
		}
		if(a[i][i]!=0)
		a[i][m]/=a[i][i];
	}
} 
```

###  数链剖分
- 路径上点权修改，动态更新 hud3966
```c++
int a[N],top[N],sz[N],dep[N],fa[N],son[N],rk[N]; //fa[i]表示树上编号i的父节点,son[i]节点i的重儿子,top[i]节点i所属重链的链头。
int n,m,cnt,c[N]; //c[i]表示树上编号i对应到线段树上的编号，rk[i]是线段树上编号i对应的树上的编号。
int sum[N<<2],add[N<<2];
void dfs1(int u,int pre,int h){
	dep[u]=h;
	fa[u]=pre;
	sz[u]=1;
	int mx=0;
	for(auto v:G[u]){
		if(v==pre)continue;
		dfs1(v,u,h+1);
		sz[u]+=sz[v];
		if(mx<sz[v])
			mx=sz[v],son[u]=v;
	}
}
void dfs2(int u,int t){
	top[u]=t;
	rk[++cnt]=u;
	c[u]=cnt;
	if(son[u]!=-1)dfs2(son[u],t);
	for(auto v:G[u]){
		if(v==fa[u]||v==son[u])continue;
		 dfs2(v,v);
	}
}
void pushUp(int rt){
	sum[rt]=sum[rt<<1]+sum[rt<<1|1];
}
void pushDown(int rt,int ln,int rn){
	if(add[rt]){
		add[rt<<1]+=add[rt];
		add[rt<<1|1]+=add[rt];
		sum[rt<<1]+=add[rt]*ln;
		sum[rt<<1|1]+=add[rt]*rn;
		add[rt]=0;
	}
}
void Build(int l,int r,int rt){
	if(l==r){
		sum[rt]=a[rk[l]];
		return ;
	}
	int mid=l+r>>1;
	if(l<=mid)Build(l,mid,rt<<1);
	if(r>mid)Build(mid+1,r,rt<<1|1);
	pushUp(rt);
}
void update(int L,int R,int c,int l,int r,int rt){
	if(L<=l&&R>=r){
		sum[rt]+=(r-l+1)*c;
		add[rt]+=c;
		return ;
	}
	int mid=l+r>>1;
	pushDown(rt,mid-l+1,r-mid);
	if(L<=mid)update(L,R,c,l,mid,rt<<1);
	if(R>mid)update(L,R,c,mid+1,r,rt<<1|1);
	pushUp(rt);
}
int query(int L,int R,int l,int r,int rt){
	if(L<=l&&R>=r){
		return sum[rt];
	}
	int mid=l+r>>1;
	int ans=0;
	pushDown(rt,mid-l+1,r-mid);
	if(L<=mid)ans+=query(L,R,l,mid,rt<<1);
	if(R>mid)ans+=query(L,R,mid+1,r,rt<<1|1);
	return ans;
}
//更新树上点x到点y路径上的点的点权。 
void solve(int x,int y,int val){
	while(top[x]!=top[y]){
		if(dep[top[x]]<dep[top[y]])swap(x,y);
		update(c[top[x]],c[x],val,1,n,1);
		x=fa[top[x]];
	}
	if(dep[x]>dep[y])swap(x,y);
	update(c[x],c[y],val,1,n,1);
}
```

### 链式前向星存图
```c++
void add_edge(int u,int v){
	edge[m].to=v;
	edge[m].nxt=head[u];
	head[u]=m++; 
}
```
### 线性基
- [线性基详解](https://blog.csdn.net/a_forever_dream/article/details/83654397)
- [线性基](https://blog.csdn.net/qaq__qaq/article/details/53812883)
```c++
struct LineBase{
	ll d[N],cnt;
	void init(){
		memset(d,0,sizeof(d));
		cnt=0;
	}
	//插入元素
	void Insert(ll x){
		for(int i=63;i>=0;--i){
			if((x>>i)&1){
				if(!d[i]){
					d[i]=x;
					cnt++;
					break;
				}
				else x^=d[i];
			}
		}
	}
	//是否可以被线性基中已有元素表示，不能就可以插入
	bool check(ll x){
		for(int i=63;i>=0;--i){
			if((x>>i)&1){
				x^=d[i];
			}
		}
		if(x)return true;
		return false;
	}
}
```
