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
- 最小点覆盖==最大匹配数 |最小点集|=|最大匹配|
- 最小路径覆盖==总点集合G-最大匹配数
-  |最大独立集| = |V|-|最大匹配数|
```
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

### st表
- 区间最大最小值 stmax[i][j] 表示i~i+2^j-1 的最大值
```
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

```
const int maxn = 10005;
int x, y, res, pre[maxn],head[maxn],n,du[maxn];
struct node {
	int to;
	int next;
}p[maxn];
int Find(int x) {
	if (pre[x] == x)return pre[x];
	return pre[x] = Find(pre[x]);
}
void tarjan(int u) {
	pre[u] = u;
	for (int i = head[u]; i != -1; i = p[i].next) {
		int v = p[i].to;
		tarjan(v);
		pre[v] = u;
	}
	if (u == x || u == y) {
		if (u != x)swap(x, y);
		if (pre[y])res = Find(pre[y]);
	}

}
//邻接表存图
scanf("%d%d", &x, &y);
			p[tot].to = y;
			p[tot].next = head[x];
			head[x] = tot++;
```

### 扩展kmp 
- 定义母串S，和字串T，设S的长度为n，T的长度为m，求T与S的每一个后缀的最长公共前缀，也就是说，设extend数组,extend[i]表示T与S[i,n-1]的最长公共前缀，要求出所有extend[i](0<=i<n)。
- 算法详解 [扩展kmp总结](https://blog.csdn.net/dyx404514/article/details/41831947)

```
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

```
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
```
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