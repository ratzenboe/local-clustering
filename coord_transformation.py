from astropy.coordinates import Galactic, ICRS, SkyCoord, GalacticLSR
from astropy import units as u
import numpy as np


### Coordinate transformation to Cartesian galactic coordinates ### 
def coord_transformation_single(ra, dec, distance, pmra_cosdec, pmdec, radial_velocity):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc,
                 pm_ra_cosdec=pmra_cosdec*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                 radial_velocity=u.Quantity(radial_velocity,u.km/u.s))
    
    gal = c.galactic  # Galactic coordinates
    
    # Cartesian Galactic heliocentric positions and velocities
    pos_hel = gal.cartesian.xyz.value  # XYZ as numpy array
    vel_hel = np.array([gal.velocity.d_x.value, gal.velocity.d_y.value, gal.velocity.d_z.value]) # VUW in km/s computed from respective RVs
    
    # Cartesian Galactic LSR velocities
    gal_lsr = gal.transform_to(GalacticLSR())  # Adds constant  (11.1, 12.24, 7.25)
    vel_lsr = np.array([gal_lsr.velocity.d_x.value, gal_lsr.velocity.d_y.value, gal_lsr.velocity.d_z.value])
    
    return c, gal, pos_hel, vel_hel, gal_lsr, vel_lsr 