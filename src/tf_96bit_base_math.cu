/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2017, 2018, 2019
                                            Oliver Weihe (o.weihe@t-online.de)
                                            George Woltman (woltman@alum.mit.edu)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/


__device__ static int cmp_ge_96(int96 a, int96 b)
/* checks if a is greater or equal than b */
{
  if(a.d2 == b.d2)
  {
    if(a.d1 == b.d1)return(a.d0 >= b.d0);
    else            return(a.d1 >  b.d1);
  }
  else              return(a.d2 >  b.d2);
}


__device__ static void shl_96(int96 *a)
/* shiftleft a one bit */
{
  a->d0 = __add_cc (a->d0, a->d0);
  a->d1 = __addc_cc(a->d1, a->d1);
  a->d2 = __addc   (a->d2, a->d2);
}


__device__ static void shl_192(int192 *a)
/* shiftleft a one bit */
{
  a->d0 = __add_cc (a->d0, a->d0);
  a->d1 = __addc_cc(a->d1, a->d1);
  a->d2 = __addc_cc(a->d2, a->d2);
  a->d3 = __addc_cc(a->d3, a->d3);
  a->d4 = __addc_cc(a->d4, a->d4);
#ifndef SHORTCUT_75BIT  
  a->d5 = __addc   (a->d5, a->d5);
#endif
}


__device__ static void sub_96(int96 *res, int96 a, int96 b)
/* a must be greater or equal b!
res = a - b */
{
  res->d0 = __sub_cc (a.d0, b.d0);
  res->d1 = __subc_cc(a.d1, b.d1);
  res->d2 = __subc   (a.d2, b.d2);
}


__device__ static void mul_96(int96 *res, int96 a, int96 b)
/* res = a * b (only lower 96 bits of the result) */
{
//#if (__CUDA_ARCH__ >= MAXWELL)
  res->d0 = __umul32  (a.d0, b.d0);

  res->d1 = __add_cc(__umul32hi(a.d0, b.d0), __umul32  (a.d1, b.d0));
  res->d2 = __addc  (__umul32  (a.d2, b.d0), __umul32hi(a.d1, b.d0));
  
  res->d1 = __add_cc(res->d1,                __umul32  (a.d0, b.d1));
  res->d2 = __addc  (res->d2,                __umul32hi(a.d0, b.d1));

  res->d2+= __umul32  (a.d0, b.d2);

  res->d2+= __umul32  (a.d1, b.d1);
}


//__device__ static void mul_96_192(int192 *res, int96 a, int96 b)
/* res = a * b */
/*{
  res->d0 = __umul32  (a.d0, b.d0);
  res->d1 = __umul32hi(a.d0, b.d0);
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d1, b.d0));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d0, b.d1));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}*/


__device__ static void mul_96_192_no_low2(int192 *res, int96 a, int96 b)
/*
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_96_192(). In
mul_96_192() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_96_192_no_low() is 0 to 2 lower than
of mul_96_192().
 */
{
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}


__device__ static void mul_96_192_no_low3(int192 *res, int96 a, int96 b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_96_192(). In
mul_96_192() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 4 lower
than of mul_96_192().
 */
{
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc   (res->d4, __umul32hi(a.d1, b.d2)); // no carry propagation to d5 needed: 0xFFFF.FFFF * 0xFFFF.FFFF + 0xFFFF.FFFF      + 0xFFFF.FFFE      = 0xFFFF.FFFF.FFFF.FFFE
                                                        //                       res->d4|d3 = (a.d1 * b.d2).hi|lo       + (a.d2 * b.d1).lo + (a.d2 * b.d0).hi

  res->d3 = __add_cc (res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (      0, __umul32hi(a.d2, b.d2));

  res->d3 = __add_cc (res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}


__device__ static void mul_96_192_no_low3_special(int192 *res, int96 a, int96 b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is partially ignored,
mul_96_192_no_low3_special differs from mul_96_192_no_low3 in that two partial
results from res.d2 are added together to generate up to one carry into res.d3.
So the digits res.d{3-5} might differ from mul_96_192(). In mul_96_192() are
three more possible carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 3 lower
than of mul_96_192().
*/
{
  unsigned int t1;

  t1      = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  t1      = __add_cc (     t1, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc   (res->d4, __umul32hi(a.d1, b.d2)); // no carry propagation to d5 needed: 0xFFFF.FFFF * 0xFFFF.FFFF + 0xFFFF.FFFF      + 0xFFFF.FFFE      + 1             = 0xFFFF.FFFF.FFFF.FFFF
                                                        //                       res->d4|d3 = (a.d1 * b.d2).hi|lo       + (a.d2 * b.d1).lo + (a.d2 * b.d0).hi + carry from t1

  res->d3 = __add_cc (res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (      0, __umul32hi(a.d2, b.d2));

  res->d3 = __add_cc (res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}


__device__ static void square_96_192(int192 *res, int96 a)
/* res = a^2
assuming that a is < 2^95 (a.d2 < 2^31)! */
{
}


__device__ static void square_96_160(int192 *res, int96 a)
/* res = a^2
this is a stripped down version of square_96_192, it doesn't compute res.d5
and is a little bit faster.
For correct results a must be less than 2^80 (a.d2 less than 2^16) */
{
}
