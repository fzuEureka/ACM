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
### Treap
```
typedef  struct TreapNode* Tree;
struct TreapNode{
    int val;
    int priority;
    Tree lchild;
    Tree rchild;
    int lsize;
    int rsize;
    TreapNode(int val=0,int priority=0){
        lchild=rchild=NULL;
        lsize=rsize=0;
        this->val=val;
        this->priority=priority;
    }
};
void left_rotate(Tree &node){
    Tree temp=node->rchild;
    node->rchild=temp->lchild;
    node->rsize=temp->lsize;
    temp->lsize=node->lsize+node->rsize+1;
    temp->lchild=node;
    node=temp;
}
void right_rotate(Tree &node){
    Tree temp=node->lchild;
    node->lchild=temp->rchild;
    node->lsize=temp->rsize;
    temp->rsize=node->lsize+node->rsize+1;
    temp->rchild=node;
    node=temp;
}
bool insert_val(Tree &root,Tree &node){
    if(root==NULL){
        root=node;
        return true;
    }
    else if(root->val<node->val){
        bool flag=insert_val(root->rchild,node);
        if(flag)root->rsize+=1;
        if(root->priority>node->priority)
            left_rotate(root);
        return  flag;
    }
    else if(root->val>node->val){
        bool flag=insert_val(root->lchild,node);
        if(flag)root->lsize+=1;
        if(root->priority>node->priority)
            right_rotate(root);
        return flag;
    }
    delete node;
    return false;
}
bool insert(Tree &root,int val,int priority){
    Tree node=new TreapNode(val,priority);
    return insert_val(root,node);
}

bool remove(Tree &root,int val)
{

    if (root->val>val) {
        root->lsize-=1;
        return remove(root->lchild, val);
    }
    else if(root->val<val) {
        root->rsize-=1;
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
        if (root->lchild->priority<root->rchild->priority){
            right_rotate(root); root->rsize-=1;
            remove(root->rchild,val);

        }
        else{
            left_rotate(root);root->lsize-=1;
            remove(root->lchild,val);

        }
    }
}
int Kth(Tree &root,int val){
    if(root->lsize==val-1)
        return root->val;
    else if(root->lsize>val-1)return Kth(root->lchild,val);
    else return Kth(root->rchild,val-(root->lsize+1));
}

Tree search(Tree &root,int val)
{
    if (!root)
        return NULL;
    else if (root->val>val)
        return search(root->lchild,val);
    else if(root->val<val)
        return  search(root->rchild,val);
    return root;
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
        if(match[i]==-1){
            memset(vis,0, sizeof(vis));
            if(dfs(i))res++;
        }
    }
    return  res;
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
