#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom_property.h"

int min_range = -1;  //no. of link list to consider in negative x
int max_range = 1;  //no. of link list to consider in positive x

//Make neighbourlist for the atoms
int makelinkedlist(a_coodrinates *input_atoms,int Natoms,float *cellsize,int *ng,long int *llst,long int ***lshd){
    int igx,igy,igz;

    for(int ii=0;ii<Natoms;ii++){
        igx = (int)(input_atoms[ii].loc[0]/cellsize[0]);
        igy = (int)(input_atoms[ii].loc[1]/cellsize[1]);
        igz = (int)(input_atoms[ii].loc[2]/cellsize[2]);
        if(igx >= ng[0] || igy >=ng[1] || igz >= ng[2]){
            fprintf(stderr,"out of list: %d %d %d %d %d %d \n",igx,igy,igz,ng[0],ng[1],ng[2]);
            exit(1);
        }
        llst[ii] = lshd[igx][igy][igz];
        lshd[igx][igy][igz]=ii;
    }
    return(0);
}

int makeneighboutlist(a_coodrinates *input_atoms,int Natoms,float *boxmd,float *halfboxmd,float *cellsize,
                      int *ng,float rcutoffsq,long int *llst,long int ***lshd,int ***retval){

    int igx,igy,igz;
    int icx,icy,icz,jj,k;
    int **nlist;
    int itype,jtype;
    double dr[3],rsq;

    nlist=(int **)malloc(Natoms*sizeof(int *));
    for(int ii=0;ii<Natoms;ii++){
        igx = (int)(input_atoms[ii].loc[0]/cellsize[0]);
        igy = (int)(input_atoms[ii].loc[1]/cellsize[1]);
        igz = (int)(input_atoms[ii].loc[2]/cellsize[2]);
        nlist[ii]=(int *)malloc((max_neighbour+1)*sizeof(int));
        nlist[ii][0]=0;
	itype = input_atoms[ii].atype;
        for(int ix=min_range; ix<=max_range; ix++){
            for(int iy=min_range; iy<=max_range; iy++){
                for(int iz=min_range; iz<=max_range; iz++){
                    icx = (igx + ix + ng[0]) % ng[0];
                    icy = (igy + iy + ng[1]) % ng[1];
                    icz = (igz + iz + ng[2]) % ng[2];
                    jj = lshd[icx][icy][icz];
                    while(jj > 0){
                        rsq=0.0;
                        if(jj != ii){
			    jtype = input_atoms[jj].atype;
			    if((itype == 1 && jtype != 1)){
                            		dr[0]=input_atoms[ii].loc[0] - input_atoms[jj].loc[0];
                            		dr[1]=input_atoms[ii].loc[1] - input_atoms[jj].loc[1];
                            		dr[2]=input_atoms[ii].loc[2] - input_atoms[jj].loc[2];
                            		for(k=0;k<3;k++){
                               	 		if (dr[k] >= halfboxmd[k]) { dr[k] -= boxmd[k]; }
                                 		else if(dr[k] <= -halfboxmd[k]) { dr[k] += boxmd[k]; }
                                 		rsq += dr[k]*dr[k];
                            		}
                            		if (rsq <= rcutoffsq) {
                                		 nlist[ii][0] +=1;
                                 		nlist[ii][nlist[ii][0]]=jj;
                            		}
			    }
                        }//end of (jj!=ii)
                        jj=llst[jj];
                    }
                }
            }
        }
        if( nlist[ii][0] > max_neighbour){
                printf("max neighbour list is larger then %d. Its %d\n",max_neighbour,nlist[ii][0]);
                exit(1);
            }
    }
    *retval=nlist;
    return(0);
}

int makeneighboutlist_peratom(a_coodrinates *input_atoms,int iatom,int Natoms,float *boxmd,float *halfboxmd,
                      float *cellsize,int *ng,float rcutoffsq,long int *llst,long int ***lshd,int **retval){

    int igx,igy,igz;
    int icx,icy,icz,ii,jj,k;
    int *nlist;
    int itype,jtype;
    double dr[3],rsq;

    ii=iatom;
    igx = (int)(input_atoms[ii].loc[0]/cellsize[0]);
    igy = (int)(input_atoms[ii].loc[1]/cellsize[1]);
    igz = (int)(input_atoms[ii].loc[2]/cellsize[2]);
    nlist=(int *)malloc((max_neighbour+1)*sizeof(int));
    nlist[0]=0;
    itype = input_atoms[ii].atype;
    for(int ix=min_range; ix<=max_range; ix++){
        for(int iy=min_range; iy<=max_range; iy++){
            for(int iz=min_range; iz<=max_range; iz++){
                icx = (igx + ix + ng[0]) % ng[0];
                icy = (igy + iy + ng[1]) % ng[1];
                icz = (igz + iz + ng[2]) % ng[2];
                jj = lshd[icx][icy][icz];
                while(jj > 0){
                    rsq=0.0;
                    if(jj != ii){
			jtype = input_atoms[jj].atype;
			if((itype == 1 && jtype != 1)){
                        	dr[0]=input_atoms[ii].loc[0] - input_atoms[jj].loc[0];
                        	dr[1]=input_atoms[ii].loc[1] - input_atoms[jj].loc[1];
                        	dr[2]=input_atoms[ii].loc[2] - input_atoms[jj].loc[2];
                        	for(k=0;k<3;k++){
                            		if (dr[k] >= halfboxmd[k]) { dr[k] -= boxmd[k]; }
                            		else if(dr[k] <= -halfboxmd[k]) { dr[k] += boxmd[k]; }
                            		rsq += dr[k]*dr[k];
                        	}
                        	if (rsq <= rcutoffsq) {
                            		nlist[0] +=1;
                            		nlist[nlist[0]]=jj;
                        	}
			}
                    }//end of (jj!=ii)
                    jj=llst[jj];
                }
            }
        }
    }
    if( nlist[0] > max_neighbour){
        printf("max neighbour list is larger then %d. Its %d\n",max_neighbour,nlist[0]);
        exit(1);
        }
    *retval=nlist;
    return(0);
}

int read_input(char *filename,a_systeminfo *mdatom_info,a_coodrinates **retval){
    char buf[1024];
    int atoms,count=-1;
    a_coodrinates *input_atoms;

    FILE *fp=fopen(filename,"r");

    //read first line to find total number of atoms
    if ( fgets(buf,sizeof(buf),fp) ){
        atoms=atoi(buf);
        mdatom_info->Natoms=atoms;
        input_atoms = (a_coodrinates *)malloc(atoms*sizeof(a_coodrinates));
    }else{
        fprintf(stderr,"First line doesnot contain number of atoms");
        fclose(fp);
        exit(1);
    }

    //read box size
    if ( fgets(buf,sizeof(buf),fp) ){
        sscanf(buf,"%f %f %f",&(mdatom_info->boxmd[0]),&(mdatom_info->boxmd[1]),&(mdatom_info->boxmd[2]));
        mdatom_info->halfboxmd[0] = 0.5*mdatom_info->boxmd[0];
        mdatom_info->halfboxmd[1] = 0.5*mdatom_info->boxmd[1];
        mdatom_info->halfboxmd[2] = 0.5*mdatom_info->boxmd[2];
    }else{
        fprintf(stderr,"Box size not defined");
        fclose(fp);
        exit(1);
    }

    //read atom coordinates and its properties
    while(fgets(buf,sizeof(buf),fp)){
        if(buf[strlen(buf)-1] !='\n'){
            fprintf(stderr, "Length of the input string is too large\n");
            exit(1);
        }
        count++;
        sscanf(buf,"%s %f %f %f",input_atoms[count].aname,&(input_atoms[count].loc[0]),
        &(input_atoms[count].loc[1]),&(input_atoms[count].loc[2]));
        if(strcmp ("Mo",input_atoms[count].aname)==0) {input_atoms[count].atype = 1;}
        else if(strcmp ("S",input_atoms[count].aname)==0) {input_atoms[count].atype = 2;}
        else if(strcmp ("O",input_atoms[count].aname)==0) {input_atoms[count].atype = 3;}
        else if(strcmp ("H",input_atoms[count].aname)==0) {input_atoms[count].atype = 4;}
        else {
           fprintf(stderr,"unknown type \n");
        }
	input_atoms[count].igroup = -1;
    }
    *retval = input_atoms;
    setsystemparameter(mdatom_info);
    return(0);
}

/*write atom coordinate, coordination number and its label*/
int write_coordinate(a_coodrinates *input_atoms,char *filename,int Natoms,float *box,int *Matom){

     FILE *fp=fopen(filename,"w");

    fprintf(fp,"%d \n",*Matom);
    fprintf(fp,"%12.6f %12.6f %12.6f \n",box[0],box[1],box[2]);
     for(int i=0;i<Natoms;i++){
        if (input_atoms[i].atype == 1 /*|| input_atoms[i].igroup == 3*/){
                fprintf(fp,"%3d \t %12.6f \t %12.6f \t %12.6f \t %6d \t %6d \n",input_atoms[i].atype,
                        input_atoms[i].loc[0],input_atoms[i].loc[1],input_atoms[i].loc[2],input_atoms[i].coordination,
                        input_atoms[i].igroup);
        }
     }
     return(0);
}

/* initilizes variable for the creation of link list*/
int setsystemparameter(a_systeminfo *mdatom_info){

    mdatom_info->binsize=BINSIZE_DEFALUT;
    mdatom_info->rcutoff=RCUTOFF_DEFAULT;
    mdatom_info->rcutoffsq=mdatom_info->rcutoff*mdatom_info->rcutoff;
    int ix,iy,iz;
    for(int ii=0;ii<3;ii++){
        mdatom_info->cellsize[ii]=mdatom_info->binsize;
        mdatom_info->ng[ii]=(int) (mdatom_info->boxmd[ii]/mdatom_info->cellsize[ii]);
        mdatom_info->cellsize[ii] = mdatom_info->boxmd[ii]/mdatom_info->ng[ii];
    }
    //create memory for linked list 
    mdatom_info->llst=(long int*)malloc(mdatom_info->Natoms*sizeof(long int));
    
    //cread memory for 3d lshd
    ix=mdatom_info->ng[0];
    iy=mdatom_info->ng[1];
    iz=mdatom_info->ng[2];
    mdatom_info->lshd=(long int***)malloc(ix*sizeof(long int**));
    for(int ii=0;ii<ix;ii++){
        mdatom_info->lshd[ii]=(long int**)malloc(iy*sizeof(long int*));
        for(int jj=0;jj<iy;jj++){
            mdatom_info->lshd[ii][jj]=(long int*)malloc(iz*sizeof(long int));
        }
    }
    //initilze all variable of lshd to -1
    for(int ii=0;ii<ix;ii++)
        for(int jj=0;jj<iy;jj++)
            for(int kk=0;kk<iz;kk++){
                mdatom_info->lshd[ii][jj][kk]=-1;
            }
    return(0);
}
