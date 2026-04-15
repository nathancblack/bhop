# Quake III Arena Movement Physics Reference

Source: [id-Software/Quake-III-Arena](https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/bg_pmove.c)

All Python implementations in `src/bhop/physics.py` must be directly traceable to this C source.

## Constants

```c
float pm_accelerate = 10.0f;      // ground acceleration
float pm_airaccelerate = 1.0f;    // air acceleration
float pm_friction = 6.0f;         // ground friction coefficient
float pm_stopspeed = 100.0f;      // speed below which friction uses stopspeed for control
float pm_waterfriction = 1.0f;    // water friction (not needed for this project)
int   sv_gravity = 800;           // gravity in units/sec^2
float sv_maxspeed = 320.0f;       // maximum wishspeed
#define JUMP_VELOCITY 270         // vertical velocity applied on jump
#define OVERCLIP 1.001            // clip velocity factor (not needed for flat plane)
```

Tick rate: 125fps competitive standard = 0.008s per frame (frametime).

## PM_Accelerate -- The Core Function

This is the function that makes bunnyhopping possible. The critical line is the dot product
projection: `currentspeed = DotProduct(pm->ps->velocity, wishdir)`.

```c
static void PM_Accelerate( vec3_t wishdir, float wishspeed, float accel ) {
    int     i;
    float   addspeed, accelspeed, currentspeed;

    currentspeed = DotProduct (pm->ps->velocity, wishdir);
    addspeed = wishspeed - currentspeed;
    if (addspeed <= 0) {
        return;
    }
    accelspeed = accel*pml.frametime*wishspeed;
    if (accelspeed > addspeed) {
        accelspeed = addspeed;
    }

    for (i=0 ; i<3 ; i++) {
        pm->ps->velocity[i] += accelspeed*wishdir[i];
    }
}
```

### Why This Enables Bunnyhopping

The bug: `currentspeed` is the **projection** of velocity onto the wish direction (dot product),
not the **magnitude** of velocity. When strafing at angle θ to your velocity:

```
currentspeed = |v| * cos(θ)
addspeed = wishspeed - |v| * cos(θ)
```

As long as `|v| * cos(θ) < wishspeed` (320), `addspeed > 0` and you gain speed.
- At 90° (perpendicular strafe): cos(90°) = 0, addspeed = 320 always → maximum acceleration
- At 45°: cos(45°) = 0.707, breaks even at |v| = 452
- At smaller angles: allows speed gain at even higher velocities

The optimal angle is: `θ_optimal = arccos(min(wishspeed / |v|, 1.0))`
- At 320 ups: any angle works
- At 640 ups: θ < ~60°
- At 1000 ups: θ < ~71°

With `pm_airaccelerate = 1.0`:
```
max accelspeed per tick = 1.0 * 0.008 * 320 = 2.56 ups
```

## PM_Friction

Only applies when on ground (`pml.walking == true`). Airborne players experience ZERO friction.

```c
static void PM_Friction( void ) {
    vec3_t  vec;
    float   *vel;
    float   speed, newspeed, control;
    float   drop;

    vel = pm->ps->velocity;

    VectorCopy( vel, vec );
    if ( pml.walking ) {
        vec[2] = 0; // ignore vertical component
    }

    speed = VectorLength( vec );
    if (speed < 1) {
        vel[0] = 0;
        vel[1] = 0; // allow sinking underwater
        return;
    }

    drop = 0;

    // apply ground friction
    if ( pm->waterlevel <= 1 ) {
        if ( pml.walking && !(pml.groundTrace.surfaceFlags & SURF_SLICK) ) {
            // if getting slow, clamp to help stop
            if ( speed < pm_stopspeed ) {
                control = pm_stopspeed;
            } else {
                control = speed;
            }
            drop += control*pm_friction*pml.frametime;
        }
    }

    // scale the velocity
    newspeed = speed - drop;
    if (newspeed < 0) {
        newspeed = 0;
    }
    newspeed /= speed;

    vel[0] *= newspeed;
    vel[1] *= newspeed;
    vel[2] *= newspeed;
}
```

## PM_CheckJump

```c
static qboolean PM_CheckJump( void ) {
    if ( pm->ps->pm_flags & PMF_RESPAWNED ) {
        return qfalse; // don't allow jump until all buttons are up
    }

    if ( pm->cmd.upmove < 10 ) {
        // not holding jump
        return qfalse;
    }

    // must wait for jump to be released
    if ( pm->ps->pm_flags & PMF_JUMP_HELD ) {
        // clear upmove so cmdscale doesn't lower running speed
        pm->cmd.upmove = 0;
        return qfalse;
    }

    pml.groundPlane = qfalse;
    pml.walking = qfalse;
    pm->ps->pm_flags |= PMF_JUMP_HELD;

    pm->ps->groundEntityNum = ENTITYNUM_NONE;
    pm->ps->velocity[2] = JUMP_VELOCITY;
    PM_AddEvent( EV_JUMP );

    if ( pm->cmd.forwardmove >= 0 ) {
        PM_ForceLegsAnim( LEGS_JUMP );
        pm->ps->pm_flags &= ~PMF_BACKWARDS_JUMP;
    } else {
        PM_ForceLegsAnim( LEGS_JUMPB );
        pm->ps->pm_flags |= PMF_BACKWARDS_JUMP;
    }

    return qtrue;
}
```

Key detail: `PMF_JUMP_HELD` -- you must **release** the jump key before you can jump again.
This means the agent must learn to release and re-press jump each time it lands.

## PM_AirMove

```c
static void PM_AirMove( void ) {
    int         i;
    vec3_t      wishvel;
    float       fmove, smove;
    vec3_t      wishdir;
    float       wishspeed;
    float       scale;
    usercmd_t   cmd;

    PM_Friction();

    fmove = pm->cmd.forwardmove;
    smove = pm->cmd.rightmove;

    cmd = pm->cmd;
    scale = PM_CmdScale( &cmd );

    // set the movementDir so clients can rotate the legs for strafing
    PM_SetMovementDir();

    // project moves down to flat plane
    pml.forward[2] = 0;
    pml.right[2] = 0;
    VectorNormalize (pml.forward);
    VectorNormalize (pml.right);

    for ( i = 0 ; i < 2 ; i++ ) {
        wishvel[i] = pml.forward[i]*fmove + pml.right[i]*smove;
    }
    wishvel[2] = 0;

    VectorCopy (wishvel, wishdir);
    wishspeed = VectorNormalize(wishdir);
    wishspeed *= scale;

    // not on ground, so little effect on velocity
    PM_Accelerate (wishdir, wishspeed, pm_airaccelerate);

    // we may have a hierarchical ground plane, step up over it
    if ( pml.groundPlane ) {
        PM_ClipVelocity (pm->ps->velocity, pml.groundTrace.plane.normal, pm->ps->velocity, OVERCLIP );
    }

    PM_StepSlideMove ( qtrue );
}
```

Note: `PM_Friction()` is called but is a no-op when airborne because `pml.walking` is false.

## PM_WalkMove

```c
static void PM_WalkMove( void ) {
    int         i;
    vec3_t      wishvel;
    float       fmove, smove;
    vec3_t      wishdir;
    float       wishspeed;
    float       scale;
    usercmd_t   cmd;
    float       accelerate;
    float       vel;

    if ( pm->waterlevel > 2 && DotProduct( pml.forward, pml.groundTrace.plane.normal ) > 0 ) {
        PM_WaterMove();
        return;
    }

    if ( PM_CheckJump () ) {
        // jumped away
        if ( pm->waterlevel > 1 ) {
            PM_WaterMove();
        } else {
            PM_AirMove();
        }
        return;
    }

    PM_Friction ();

    fmove = pm->cmd.forwardmove;
    smove = pm->cmd.rightmove;

    cmd = pm->cmd;
    scale = PM_CmdScale( &cmd );

    // set the movementDir so clients can rotate the legs for strafing
    PM_SetMovementDir();

    // project moves down to flat plane
    pml.forward[2] = 0;
    pml.right[2] = 0;

    // project the forward and right directions onto the ground plane
    PM_ClipVelocity (pml.forward, pml.groundTrace.plane.normal, pml.forward, OVERCLIP );
    PM_ClipVelocity (pml.right, pml.groundTrace.plane.normal, pml.right, OVERCLIP );

    VectorNormalize (pml.forward);
    VectorNormalize (pml.right);

    for ( i = 0 ; i < 2 ; i++ ) {
        wishvel[i] = pml.forward[i]*fmove + pml.right[i]*smove;
    }
    wishvel[2] = 0;

    VectorCopy (wishvel, wishdir);
    wishspeed = VectorNormalize(wishdir);
    wishspeed *= scale;

    // clamp the speed lower if ducking
    if ( pm->ps->pm_flags & PMF_DUCKED ) {
        if ( wishspeed > pm->ps->speed * pm_duckScale ) {
            wishspeed = pm->ps->speed * pm_duckScale;
        }
    }

    // clamp the speed lower if wading or walking on the bottom
    if ( pm->waterlevel ) {
        float waterScale;
        waterScale = pm->waterlevel / 3.0;
        waterScale = 1.0 - ( 1.0 - pm_swimScale ) * waterScale;
        if ( wishspeed > pm->ps->speed * waterScale ) {
            wishspeed = pm->ps->speed * waterScale;
        }
    }

    accelerate = pm_accelerate;

    PM_Accelerate (wishdir, wishspeed, accelerate);

    // ... (slide move, step handling)
}
```

Key: `PM_CheckJump()` is called FIRST in WalkMove. If it returns true, execution immediately
switches to `PM_AirMove()`. This means on the frame you jump, you get air physics (no ground friction applied for that tick).

## PM_CmdScale -- Diagonal Normalization

Prevents forward+strafe from being sqrt(2) faster than forward alone.

```c
static float PM_CmdScale( usercmd_t *cmd ) {
    int     max;
    float   total;
    float   scale;

    max = abs( cmd->forwardmove );
    if ( abs( cmd->rightmove ) > max ) {
        max = abs( cmd->rightmove );
    }
    if ( abs( cmd->upmove ) > max ) {
        max = abs( cmd->upmove );
    }
    if ( !max ) {
        return 0;
    }

    total = sqrt( cmd->forwardmove * cmd->forwardmove
        + cmd->rightmove * cmd->rightmove + cmd->upmove * cmd->upmove );
    scale = pm->ps->speed * max / ( 127.0 * total );

    return scale;
}
```

Note: `pm->ps->speed` is the player's `sv_maxspeed` (320).

## Gravity

Applied as half-steps in the main movement loop:

```c
// Before movement:
pm->ps->velocity[2] -= pm->ps->gravity * pml.frametime * 0.5;
// After movement:
pm->ps->velocity[2] -= pm->ps->gravity * pml.frametime * 0.5;
```

This is standard Verlet integration. `pm->ps->gravity` = `sv_gravity` = 800.

## Ground Trace (Simplified for Flat Plane)

In full Q3, this does a BSP trace to find the ground surface. For our flat infinite plane:
- Ground is at z = 0
- Player is on ground if position.z <= 0 AND velocity.z <= 0
- When landing: set position.z = 0, velocity.z = 0, on_ground = true

## Coordinate System

Quake III uses Z-up (X = forward, Y = left, Z = up).
The Python implementation should use the same convention for consistency with the source.

## Key Resources

- [Quake III Arena source](https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/bg_pmove.c)
- [Bunnyhopping from the Programmer's Perspective](https://adrianb.io/2015/02/14/bunnyhop.html)
- [QuakeWorld Air Physics](https://www.quakeworld.nu/wiki/QW_physics_air)
- [WiggleWizard/quake3-movement-unity3d](https://github.com/WiggleWizard/quake3-movement-unity3d)
- [Recreating Quake Movement in Godot 4.0](https://aneacsu.com/blog/2023-04-09-quake-movement-godot)
- [q3df-doc DeFRaG physics docs](https://github.com/Jelvan1/q3df-doc)
