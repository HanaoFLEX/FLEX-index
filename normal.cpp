#include<iostream>
#include<cstring>
#include<algorithm>
#include<fstream>
#include<ctime>
#include<cstdlib>
#include<random>
#include<vector>

using namespace std;
mt19937 gen((unsigned int) time(nullptr));

int main()
{
    int n,dim,min_mu,max_mu,min_sigma,max_sigma;
    cout<<"plz input n dim min_mu max_mu min_sigma max_sigma: ";
    cin>>n>>dim>>min_mu>>max_mu>>min_sigma>>max_sigma;
    cout<<"start construction"<<endl;
    string file_name=string("normal_n")+to_string(n)+"_dim"+to_string(dim)+"_mnmu"+to_string(min_mu)+"_mxmu"+
                to_string(max_mu)+"_mnsig"+to_string(min_sigma)+"_mxsig"+to_string(max_sigma)+".csv";
    ofstream ouf(file_name,ios::out);
    vector<normal_distribution<double>> normals;
    uniform_int_distribution<int> gu(min_mu,max_mu),gs(min_sigma,max_sigma);
    for(int i=0;i<dim;i++)
        normals.push_back(normal_distribution<double>(gu(gen),gs(gen)));
    for(int i=0;i<n;i++)
    {
        if(i and i%(n/10)==0)
        {
            printf("now construct %.2f%%\n",100.0*i/n);
        }
        for(int j=0;j<dim;j++)
        {
            if(j) ouf<<",";
            ouf<<normals[j](gen);
        }
        ouf<<endl;
    }
    ouf.close();
    return 0;
}
