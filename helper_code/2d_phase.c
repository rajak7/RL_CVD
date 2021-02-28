#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom_property.h"

int cal_distance2d(int iatom, int jatom,a_coodrinates *atoms,float *boxmd,float *halfboxmd,float *retval){
    float rsq=0.0;
    float dr[2]={0.0,0.0};

    for(int kk=0;kk<2;kk++){
	dr[kk] = atoms[iatom].loc[kk] - atoms[jatom].loc[kk];
	if (dr[kk] > halfboxmd[kk]) {dr[kk] -= boxmd[kk];}
	if (dr[kk] < -halfboxmd[kk]) {dr[kk] += boxmd[kk];}
	rsq+=dr[kk]*dr[kk];
	retval[kk]=dr[kk];
    }
    retval[2]=sqrt(rsq);
    return(0);
}

int cmpfunc (const void * a, const void * b) {
     return ( *(float*)a - *(float*)b );
}

int cal_angle(a_coodrinates *atoms,int iatom,int *nlist,float *boxmd,float *halfboxmd){

    int jatom,katom,count=0;
    int N_top=0,N_bottom=0;
    int toplist[3],botlist[3];
    float dis2d_ij[3],dis2d_ik[3];
    float angleval[9],costheta_ijk,top_val;
    float threshold = 20.0;

    if (nlist[0] == 6){

	 //calculate top and bottom list
	 for(int ii=1;ii<=nlist[0];ii++){
            jatom = nlist[ii];
	    if (atoms[jatom].atype != 2) 
	    	return 3;
            atoms[jatom].igroup = 3;
	    if(atoms[jatom].loc[2] > atoms[iatom].loc[2]){
	        N_top +=1;
		if(N_top >3){ return 3; }
		toplist[N_top-1] = jatom;
	    }else{
	        N_bottom+=1;
		if(N_bottom >3){ return 3; }
		botlist[N_bottom-1] = jatom;
	    }
	 }
	 // calculate angle matrix
	 for(int ii=0;ii<3;ii++){
	    jatom = toplist[ii];
	    cal_distance2d(iatom,jatom,atoms,boxmd,halfboxmd,dis2d_ij);
	    for(int jj=0;jj<3;jj++){
		katom = botlist[jj];
		cal_distance2d(iatom,katom,atoms,boxmd,halfboxmd,dis2d_ik);
		top_val = dis2d_ij[0]*dis2d_ik[0] + dis2d_ij[1]*dis2d_ik[1];
		costheta_ijk = top_val/(dis2d_ij[2]*dis2d_ik[2]);
		if(costheta_ijk <= -1.00) {costheta_ijk += 0.001;}
		if (costheta_ijk >= 1.00) {costheta_ijk -= 0.001;}
		angleval[count++] = 180.0*acos(costheta_ijk)/3.14;
	    }
	 }
	 qsort(angleval,sizeof(angleval)/sizeof(angleval[0]),sizeof(angleval[0]),cmpfunc);
	 if (angleval[0]+angleval[1]+angleval[2] < threshold) {
		 return 1;  //2H
	 }else {
		 return 2; //1T
	 }
    }else{
         return 3;  //defect
    }
}

int write_tensor(a_coodrinates *input_atoms,char *f, char *filename,int Natoms,float *box,int *Matom){

        float **tensor_mo;
        int ngrids = 50, x_loc, y_loc;
        float delx, dely;
        FILE *fp=fopen(filename,"w");
        FILE *fp1 = fopen(f, "w");

        delx = box[0]/((float)ngrids);
        dely = box[1]/((float)ngrids);
        fprintf(stdout,"grid size %f %f  \n",delx,dely);

        tensor_mo=(float **)malloc(ngrids*sizeof(float *));

        for(int ii = 0; ii < ngrids; ii++){
                 tensor_mo[ii]=(float *)malloc(ngrids*sizeof(float));
                 for(int jj=0;jj<ngrids;jj++){
                         tensor_mo[ii][jj] = 0.0;
                 }
        }

        fprintf(fp1,"%d \n",*Matom);
	fprintf(fp1,"%12.6f %12.6f %12.6f \n",box[0],box[1],box[2]);

        for(int i=0;i<Natoms;i++){
                if (input_atoms[i].atype == 1){
                        x_loc = (int)(input_atoms[i].loc[0]/delx);
                        y_loc = (int)(input_atoms[i].loc[1]/dely);
			//  fprintf(stdout,"grid val %d %d %d\n",x_loc,y_loc,input_atoms[i].igroup);
                        if (x_loc >= ngrids || y_loc >= ngrids){
                                fprintf(stderr,"x_loc or y_loc out of bound % d %d \n",x_loc,y_loc);
                                exit(1);
                        }
			//printf("%d   %d   %6d\n" , x_loc, y_loc,input_atoms[i].igroup);
                        tensor_mo[x_loc][y_loc] = (float)input_atoms[i].igroup;
                
			fprintf(fp1,"%3d \t %12.6f \t %12.6f \t %12.6f \t %6d \t %6d \t %6d\n",input_atoms[i].atype, 
			input_atoms[i].loc[0],input_atoms[i].loc[1],input_atoms[i].loc[2],x_loc, y_loc, input_atoms[i].igroup);	
		}
	}

        fprintf(fp,"%d \n",ngrids);
        for(int ii = 0; ii < ngrids; ii++){
                 for(int jj=0;jj<ngrids;jj++){
                         fprintf(fp,"%12.6f ",tensor_mo[ii][jj]);
                 }
                 fprintf(fp,"\n");
        }
        fclose(fp);
        return(0);

}
int main(int argc, char* argv[]){
    char *filename,*out_filename;
    a_systeminfo mdatom_info;
    a_coodrinates *input_atoms;
    int *nlist,Matom=0;

    filename=argv[1];
    out_filename  = argv[2];
    read_input(filename,&mdatom_info,&input_atoms);
    makelinkedlist(input_atoms,mdatom_info.Natoms,mdatom_info.cellsize,mdatom_info.ng,mdatom_info.llst,mdatom_info.lshd);
    fprintf(stdout,"Total number of atoms %10d \n",mdatom_info.Natoms);
    fprintf(stdout,"Box size %12.6f %12.6f %12.6f \n",mdatom_info.boxmd[0],mdatom_info.boxmd[1],mdatom_info.boxmd[2]);

    //feature vector

    for(int i=0;i<mdatom_info.Natoms;i++){
	if (input_atoms[i].atype == 1){
                Matom+=1;
		makeneighboutlist_peratom(input_atoms,i,mdatom_info.Natoms,mdatom_info.boxmd,mdatom_info.halfboxmd,
				mdatom_info.cellsize,mdatom_info.ng,mdatom_info.rcutoffsq,
				mdatom_info.llst,mdatom_info.lshd,&nlist);
		input_atoms[i].coordination = nlist[0];
		input_atoms[i].igroup = cal_angle(input_atoms,i,nlist,mdatom_info.boxmd,mdatom_info.halfboxmd);
		free(nlist);
	}
    }
    //write_coordinate(input_atoms,filename,mdatom_info.Natoms,mdatom_info.boxmd,&Matom);
    write_tensor(input_atoms,filename, out_filename,mdatom_info.Natoms,mdatom_info.boxmd,&Matom);
    return(0);
}
