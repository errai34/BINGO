
#
# Read Andrea's APOKASC,GDR2 data
# also output the centre of the field
# with MC error estimates
#

import pyfits
import numpy
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from galpy.util import bovy_coords
from galpy.util import bovy_plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from StringIO import StringIO
from mpi4py import MPI

comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

# read fits data
# star_hdus=pyfits.open('../ARICH_APR18_R7_APOKASC_v3.6.5_GDR2.fits')
star_hdus=pyfits.open('../data13Jul18/7dotS26.fits')
star=star_hdus[1].data
star_hdus.close()

# number of stars
# nstar=len(star['KEPLER_ID'])
nstar=len(star['KIC'])
if myrank==0:
    print 'total number of start=',nstar

sindx = numpy.where((star['dist']>0.0) & (star['radial_velocity_error']>0.0))
# nstar=len(star['KEPLER_ID'][sindx])
nstar=len(star['KIC'][sindx])
if myrank==0:
    print 'number of start (d>0) =',nstar

# stellar data info
ra = star['ra_gaia'][sindx]
dec = star['dec_gaia'][sindx]
pmra=star['pmra'][sindx]
pmra_err=star['pmra_error'][sindx]
pmdec=star['pmdec'][sindx]
pmdec_err=star['pmdec_error'][sindx]
vhel=star['radial_velocity'][sindx]
vhel_err=star['radial_velocity_error'][sindx]
# unit kpc, seismic distance
dist=star['dist'][sindx]/1000.0
# dist_68l=star['dist_68L_1'][sindx]/1000.0
# dist_68u=star['dist_68U_1'][sindx]/1000.0
dist_68l=star['dist_68L'][sindx]/1000.0
dist_68u=star['dist_68U'][sindx]/1000.0
dist_err = 0.5*((dist-dist_68l)+(dist_68u-dist))/1000.0
dist_err_min = 0.000001
if myrank == 0:
    print ' N(d>derr)=',len(dist_err[dist_err<dist_err_min])
dist_err[dist_err<dist_err_min] = dist_err_min
# other properties
# June Y18 version
# age = star['age_1'][sindx]
# age_68l = star['age_68L_1'][sindx]
# age_68u = star['age_68U_1'][sindx]
# feh = star['feh_1'][sindx]
# alpha = star['alpha'][sindx]
# data13Jul18
age = star['age'][sindx]
age_68l = star['age_68L'][sindx]
age_68u = star['age_68U'][sindx]
feh = star['FE_H_ADOP_COR'][sindx]
alpha = star['ALP_FE_ADOP_COR'][sindx]

# convert coordinates
Tllbb=bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=None)
glon=Tllbb[:,0]
glat=Tllbb[:,1]

# assumed solar motion, from Adibekyan et al. (2012)
usun=11.1
vsun=12.24
wsun=7.25
zsun=0.014

# set vxvv for galpy
# vxvs=[]
# for i in range(0,nstar-1):
#  vxvs.append([ra[i],dec[i],dist[i],pmra[i],pmdec[i],vhel[i]])
# vxvv=numpy.array(vxvs)

# measure Rm, zmax and ecc
rap=numpy.zeros(nstar,dtype=numpy.float64)
rap_err=numpy.zeros(nstar,dtype=numpy.float64)
rperi=numpy.zeros(nstar,dtype=numpy.float64)
rperi_err=numpy.zeros(nstar,dtype=numpy.float64)
rmean=numpy.zeros(nstar,dtype=numpy.float64)
rmean_err=numpy.zeros(nstar,dtype=numpy.float64)
zmax=numpy.zeros(nstar,dtype=numpy.float64)
zmax_err=numpy.zeros(nstar,dtype=numpy.float64)
ecc=numpy.zeros(nstar,dtype=numpy.float64)
ecc_err=numpy.zeros(nstar,dtype=numpy.float64)
vphi=numpy.zeros(nstar,dtype=numpy.float64)
vphi_err=numpy.zeros(nstar,dtype=numpy.float64)
vrad=numpy.zeros(nstar,dtype=numpy.float64)
vrad_err=numpy.zeros(nstar,dtype=numpy.float64)
ene=numpy.zeros(nstar,dtype=numpy.float64)
ene_err=numpy.zeros(nstar,dtype=numpy.float64)
lz=numpy.zeros(nstar,dtype=numpy.float64)
lz_err=numpy.zeros(nstar,dtype=numpy.float64)
# time steps
# ts= numpy.linspace(0,100,10000)
# for test
ts= numpy.linspace(0,10,100)

# number of MC
# for test
# nmc=1000
nmc=10
# MC sampled values
rap_mc=numpy.zeros(nmc)
rperi_mc=numpy.zeros(nmc)
rmean_mc=numpy.zeros(nmc)
zmax_mc=numpy.zeros(nmc)
ecc_mc=numpy.zeros(nmc)
vphi_mc=numpy.zeros(nmc)
vrad_mc=numpy.zeros(nmc)
ene_mc=numpy.zeros(nmc)
lz_mc=numpy.zeros(nmc)

for i in range(myrank,nstar,nprocs):
    dist_mc=numpy.random.normal(dist[i],dist_err[i],nmc)
    pmra_mc=numpy.random.normal(pmra[i],pmra_err[i],nmc)
    pmdec_mc=numpy.random.normal(pmdec[i],pmdec_err[i],nmc)
    vhel_mc=numpy.random.normal(vhel[i],vhel_err[i],nmc)
    if len(dist_mc[dist_mc<0])>0:
        print ' star',i,' N(dist<0)=',len(dist_mc[dist_mc<0]),' set to 0 for all'
        pmra_mc[dist_mc<0]=0.0
        vhel_mc[dist_mc<0]=0.0
        pmdec_mc[dist_mc<0]=0.0
        dist_mc[dist_mc<0]=0.0

    for j in range(nmc):
        op= Orbit(vxvv=[ra[i],dec[i],dist_mc[j],pmra_mc[j],pmdec_mc[j],vhel_mc[j]],\
          radec=True,uvw=False,ro=8.0, vo=220.0,zo=zsun, \
          solarmotion=[-usun,vsun,wsun])
        op.integrate(ts,MWPotential2014)
        rap_mc[j]=op.rap()
        rperi_mc[j]=op.rperi()
        rmean_mc[j]=0.5*(rap_mc[j]+rperi_mc[j])
        zmax_mc[j]=op.zmax()
        ecc_mc[j]=op.e()
        ene_mc[j]=op.E()
        lz_mc[j]=op.L(0.0)[0][2]
        vphi_mc[j]=op.vphi(0.0)
        vrad_mc[j]=op.vR(0.0)
    # taking mean and std
    rap[i]=rap_mc.mean()
    rap_err[i]=rap_mc.std()
    rperi[i]=rperi_mc.mean()
    rperi_err[i]=rperi_mc.std()
    rmean[i]=rmean_mc.mean()
    rmean_err[i]=rmean_mc.std()
    zmax[i]=zmax_mc.mean()
    zmax_err[i]=zmax_mc.std()
    ecc[i]=ecc_mc.mean()
    ecc_err[i]=ecc_mc.std()
    ene[i]=ene_mc.mean()
    ene_err[i]=ene_mc.std()
    lz[i]=lz_mc.mean()
    lz_err[i]=lz_mc.std()
    vphi[i]=vphi_mc.mean()
    vphi_err[i]=vphi_mc.std()
    vrad[i]=vrad_mc.mean()
    vrad_err[i]=vrad_mc.std()

# MPI 
sendbuf=numpy.zeros(nstar,dtype=numpy.float64)
# Rap
sendbuf=rap
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rap=recvbuf
# error
sendbuf=rap_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rap_err=recvbuf
# Rperi
sendbuf=rperi
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rperi=recvbuf
# err
sendbuf=rperi_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rperi_err=recvbuf
# Rmean
sendbuf=rmean
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rmean=recvbuf
# err
sendbuf=rmean_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
rmean_err=recvbuf
# zmax
sendbuf=zmax
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
zmax=recvbuf
# err
sendbuf=zmax_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
zmax_err=recvbuf
# ecc
sendbuf=ecc
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
ecc=recvbuf
# err
sendbuf=ecc_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
ecc_err=recvbuf
# ene
sendbuf=ene
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
ene=recvbuf
# err
sendbuf=ene_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
ene_err=recvbuf
# lz
sendbuf=lz
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
lz=recvbuf
# err
sendbuf=lz_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
lz_err=recvbuf
# vphi
sendbuf=vphi
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
vphi=recvbuf
# err
sendbuf=vphi_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
vphi_err=recvbuf
# vrad
sendbuf=vrad
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
vrad=recvbuf
# err
sendbuf=vrad_err
recvbuf=numpy.zeros(nstar,dtype=numpy.float64)
comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
vrad_err=recvbuf

if myrank==0:
  f=open('arich-orb.asc','w')
  for i in range(0,nstar-1):
    print >>f, "%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f" %(\
      glon[i],glat[i],dist[i],feh[i],alpha[i],age[i],rap[i],rperi[i],rmean[i] \
      ,zmax[i],ecc[i],ene[i],lz[i],vphi[i],vrad[i],ra[i],dec[i],vhel[i])
  f.close()

# Fits output
# output fits file
#  tbhdu = pyfits.BinTableHDU.from_columns([\
#    pyfits.Column(name='KIC',format='J',array=star['KEPLER_ID'][sindx]),\
  tbhdu = pyfits.BinTableHDU.from_columns([\
    pyfits.Column(name='KIC',format='J',array=star['KIC'][sindx]),\
    pyfits.Column(name='Glon',format='D',array=glon),\
    pyfits.Column(name='Glat',format='D',array=glat),\
    pyfits.Column(name='RA',format='D',array=ra),\
    pyfits.Column(name='DEC',format='D',array=dec),\
    pyfits.Column(name='PMRA',format='D',array=pmra),\
    pyfits.Column(name='PMDEC',format='D',array=pmdec),\
    pyfits.Column(name='PMRA_err',format='D',array=pmra_err),\
    pyfits.Column(name='PMDEC_err',format='D',array=pmdec_err),\
    # pyfits.Column(name='Teff',format='D',array=star['teff_1'][sindx]),\
    # pyfits.Column(name='Teff_err',format='D',array=star['eteff'][sindx]),\
    pyfits.Column(name='Fe_H',format='D',array=feh),\
    pyfits.Column(name='Alpha',format='D',array=alpha),\
    pyfits.Column(name='Vhelio',format='D',array=vhel),\
    pyfits.Column(name='Vhelio_err',format='D',array=vhel_err),\
    pyfits.Column(name='Age',format='D',array=age),\
    pyfits.Column(name='Age_68L',format='D',array=age_68l),\
    pyfits.Column(name='Age_68U',format='D',array=age_68u),\
    pyfits.Column(name='Dist',format='D',array=dist),\
    # pyfits.Column(name='Dist_68L',format='D',array=star['Dist_68L_1'][sindx]),\
    # pyfits.Column(name='Dist_68U',format='D',array=star['Dist_68U_1'][sindx]),\
    pyfits.Column(name='Dist_68L',format='D',array=star['dist_68L'][sindx]),\
    pyfits.Column(name='Dist_68U',format='D',array=star['dist_68U'][sindx]),
    pyfits.Column(name='RM',format='D',array=rmean),\
    pyfits.Column(name='RM_err',format='D',array=rmean_err),\
    pyfits.Column(name='ECC',format='D',array=ecc),\
    pyfits.Column(name='ECC_err',format='D',array=ecc_err),\
    pyfits.Column(name='ZMAX',format='D',array=zmax),\
    pyfits.Column(name='ZMAX_err',format='D',array=zmax_err),\
    pyfits.Column(name='Rperi',format='D',array=rperi),\
    pyfits.Column(name='Rperi_err',format='D',array=rperi_err),\
    pyfits.Column(name='Rapo',format='D',array=rap),\
    pyfits.Column(name='Rapo_err',format='D',array=rap_err),\
    pyfits.Column(name='E',format='D',array=ene),\
    pyfits.Column(name='E_err',format='D',array=ene_err),\
    pyfits.Column(name='LZ',format='D',array=lz),\
    pyfits.Column(name='LZ_err',format='D',array=lz_err),\
    pyfits.Column(name='VPHI',format='D',array=vphi),\
    pyfits.Column(name='VPHI_err',format='D',array=vphi_err),\
    pyfits.Column(name='VR',format='D',array=vrad),\
    pyfits.Column(name='VR_err',format='D',array=vrad_err)])
  tbhdu.writeto('arich-orb.fits',clobber=True)

  plt.plot(rmean,zmax,"o")
  plt.errorbar(rmean,zmax,xerr=rmean_err,yerr=zmax_err,fmt='o')
  plt.xlabel(r"R$_{\rm mean}$",fontsize=12,fontname="serif")
  plt.ylabel(r"z$_{\rm max}$",fontsize=12,fontname="serif")
  plt.axis([0.0,15.0,0.0,15.0])
  plt.show()

comm.Disconnect()




