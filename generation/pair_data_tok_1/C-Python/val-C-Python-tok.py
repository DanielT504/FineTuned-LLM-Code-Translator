def Alphabet_N_Pattern ( N ) : NEW_LINE
Right = 1 NEW_LINE Left = 1 NEW_LINE Diagonal = 2 NEW_LINE
for index in range ( N ) : NEW_LINE
print ( Left , end = " " ) NEW_LINE Left += 1 NEW_LINE
for side_index in range ( 0 , 2 * ( index ) , 1 ) : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
if ( index != 0 and index != N - 1 ) : NEW_LINE INDENT print ( Diagonal , end = " " ) NEW_LINE Diagonal += 1 NEW_LINE DEDENT else : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
for side_index in range ( 0 , 2 * ( N - index - 1 ) , 1 ) : NEW_LINE INDENT print ( " ▁ " , end = " " ) NEW_LINE DEDENT
print ( Right , end = " " ) NEW_LINE Right += 1 NEW_LINE print ( " " , ▁ end ▁ = ▁ " " ) NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
Size = 6 NEW_LINE
Alphabet_N_Pattern ( Size ) NEW_LINE
def permutationCoeff ( n , k ) : NEW_LINE INDENT P = [ [ 0 for i in range ( k + 1 ) ] for j in range ( n + 1 ) ] NEW_LINE DEDENT
for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( min ( i , k ) + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT P [ i ] [ j ] = 1 NEW_LINE DEDENT
else : NEW_LINE INDENT P [ i ] [ j ] = P [ i - 1 ] [ j ] + ( j * P [ i - 1 ] [ j - 1 ] ) NEW_LINE DEDENT
if ( j < k ) : NEW_LINE INDENT P [ i ] [ j + 1 ] = 0 NEW_LINE DEDENT return P [ n ] [ k ] NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ fo ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) , sep = " " ) NEW_LINE
def permutationCoeff ( n , k ) : NEW_LINE INDENT fact = [ 0 for i in range ( n + 1 ) ] NEW_LINE DEDENT
fact [ 0 ] = 1 NEW_LINE
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT fact [ i ] = i * fact [ i - 1 ] NEW_LINE DEDENT
return int ( fact [ n ] / fact [ n - k ] ) NEW_LINE
n = 10 NEW_LINE k = 2 NEW_LINE print ( " Value ▁ of ▁ P ( " , n , " , ▁ " , k , " ) ▁ is ▁ " , permutationCoeff ( n , k ) , sep = " " ) NEW_LINE
def isSubsetSum ( set , n , sum ) : NEW_LINE
if ( sum == 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT if ( n == 0 ) : NEW_LINE INDENT return False NEW_LINE DEDENT
if ( set [ n - 1 ] > sum ) : NEW_LINE INDENT return isSubsetSum ( set , n - 1 , sum ) NEW_LINE DEDENT
return isSubsetSum ( set , n - 1 , sum ) or isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) NEW_LINE
set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
def pell ( n ) : NEW_LINE INDENT if ( n <= 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT a = 1 NEW_LINE b = 2 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE INDENT c = 2 * b + a NEW_LINE a = b NEW_LINE b = c NEW_LINE DEDENT return b NEW_LINE DEDENT
n = 4 NEW_LINE print ( pell ( n ) ) NEW_LINE
def factorial ( n ) : NEW_LINE INDENT if n == 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT return n * factorial ( n - 1 ) NEW_LINE DEDENT
num = 5 ; NEW_LINE print ( " Factorial ▁ of " , num , " is " , factorial ( num ) ) NEW_LINE
def findSubArray ( arr , n ) : NEW_LINE INDENT sum = 0 NEW_LINE maxsize = - 1 NEW_LINE DEDENT
for i in range ( 0 , n - 1 ) : NEW_LINE INDENT sum = - 1 if ( arr [ i ] == 0 ) else 1 NEW_LINE DEDENT
for j in range ( i + 1 , n ) : NEW_LINE INDENT sum = sum + ( - 1 ) if ( arr [ j ] == 0 ) else sum + 1 NEW_LINE DEDENT
if ( sum == 0 and maxsize < j - i + 1 ) : NEW_LINE INDENT maxsize = j - i + 1 NEW_LINE startindex = i NEW_LINE DEDENT if ( maxsize == - 1 ) : NEW_LINE print ( " No ▁ such ▁ subarray " ) ; NEW_LINE else : NEW_LINE print ( startindex , " to " , startindex + maxsize - 1 ) ; NEW_LINE return maxsize NEW_LINE
arr = [ 1 , 0 , 0 , 1 , 0 , 1 , 1 ] NEW_LINE size = len ( arr ) NEW_LINE findSubArray ( arr , size ) NEW_LINE
def ternarySearch ( l , r , key , ar ) : NEW_LINE INDENT while r >= l : NEW_LINE DEDENT
mid1 = l + ( r - l ) // 3 NEW_LINE mid2 = r - ( r - l ) // 3 NEW_LINE
if key == ar [ mid1 ] : NEW_LINE INDENT return mid1 NEW_LINE DEDENT if key == mid2 : NEW_LINE INDENT return mid2 NEW_LINE DEDENT
if key < ar [ mid1 ] : NEW_LINE
r = mid1 - 1 NEW_LINE elif key > ar [ mid2 ] : NEW_LINE
l = mid2 + 1 NEW_LINE else : NEW_LINE
l = mid1 + 1 NEW_LINE r = mid2 - 1 NEW_LINE
return - 1 NEW_LINE
ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE
l = 0 NEW_LINE
r = 9 NEW_LINE
key = 5 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
key = 50 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
def findMin ( arr , low , high ) : NEW_LINE
if high < low : NEW_LINE INDENT return arr [ 0 ] NEW_LINE DEDENT
if high == low : NEW_LINE INDENT return arr [ low ] NEW_LINE DEDENT
mid = int ( ( low + high ) / 2 ) NEW_LINE
if mid < high and arr [ mid + 1 ] < arr [ mid ] : NEW_LINE INDENT return arr [ mid + 1 ] NEW_LINE DEDENT
if mid > low and arr [ mid ] < arr [ mid - 1 ] : NEW_LINE INDENT return arr [ mid ] NEW_LINE DEDENT
if arr [ high ] > arr [ mid ] : NEW_LINE INDENT return findMin ( arr , low , mid - 1 ) NEW_LINE DEDENT return findMin ( arr , mid + 1 , high ) NEW_LINE
arr1 = [ 5 , 6 , 1 , 2 , 3 , 4 ] NEW_LINE n1 = len ( arr1 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr1 , 0 , n1 - 1 ) ) ) NEW_LINE arr2 = [ 1 , 2 , 3 , 4 ] NEW_LINE n2 = len ( arr2 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr2 , 0 , n2 - 1 ) ) ) NEW_LINE arr3 = [ 1 ] NEW_LINE n3 = len ( arr3 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr3 , 0 , n3 - 1 ) ) ) NEW_LINE arr4 = [ 1 , 2 ] NEW_LINE n4 = len ( arr4 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr4 , 0 , n4 - 1 ) ) ) NEW_LINE arr5 = [ 2 , 1 ] NEW_LINE n5 = len ( arr5 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr5 , 0 , n5 - 1 ) ) ) NEW_LINE arr6 = [ 5 , 6 , 7 , 1 , 2 , 3 , 4 ] NEW_LINE n6 = len ( arr6 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr6 , 0 , n6 - 1 ) ) ) NEW_LINE arr7 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] NEW_LINE n7 = len ( arr7 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr7 , 0 , n7 - 1 ) ) ) NEW_LINE arr8 = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ] NEW_LINE n8 = len ( arr8 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr8 , 0 , n8 - 1 ) ) ) NEW_LINE arr9 = [ 3 , 4 , 5 , 1 , 2 ] NEW_LINE n9 = len ( arr9 ) NEW_LINE print ( " The ▁ minimum ▁ element ▁ is ▁ " + str ( findMin ( arr9 , 0 , n9 - 1 ) ) ) NEW_LINE
import sys NEW_LINE
def print2Smallest ( arr ) : NEW_LINE
arr_size = len ( arr ) NEW_LINE if arr_size < 2 : NEW_LINE INDENT print " Invalid ▁ Input " NEW_LINE return NEW_LINE DEDENT first = second = sys . maxint NEW_LINE for i in range ( 0 , arr_size ) : NEW_LINE
if arr [ i ] < first : NEW_LINE INDENT second = first NEW_LINE first = arr [ i ] NEW_LINE DEDENT
elif ( arr [ i ] < second and arr [ i ] != first ) : NEW_LINE INDENT second = arr [ i ] ; NEW_LINE DEDENT if ( second == sys . maxint ) : NEW_LINE print " No ▁ second ▁ smallest ▁ element " NEW_LINE else : NEW_LINE print ' The ▁ smallest ▁ element ▁ is ' , first , ' and ' ' ▁ second ▁ smallest ▁ element ▁ is ' , second NEW_LINE
arr = [ 12 , 13 , 1 , 10 , 34 , 1 ] NEW_LINE print2Smallest ( arr ) NEW_LINE
def isSubsetSum ( arr , n , sum ) : NEW_LINE
subset = [ [ False for j in range ( sum + 1 ) ] for i in range ( 3 ) ] NEW_LINE for i in range ( n + 1 ) : NEW_LINE INDENT for j in range ( sum + 1 ) : NEW_LINE DEDENT
if ( j == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = True NEW_LINE DEDENT
elif ( i == 0 ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = False NEW_LINE DEDENT elif ( arr [ i - 1 ] <= j ) : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] or subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT else : NEW_LINE INDENT subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] NEW_LINE DEDENT return subset [ n % 2 ] [ sum ] NEW_LINE
arr = [ 6 , 2 , 5 ] NEW_LINE sum = 7 NEW_LINE n = len ( arr ) NEW_LINE if ( isSubsetSum ( arr , n , sum ) == True ) : NEW_LINE INDENT print ( " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT
def findCandidate ( A ) : NEW_LINE INDENT maj_index = 0 NEW_LINE count = 1 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE INDENT if A [ maj_index ] == A [ i ] : NEW_LINE INDENT count += 1 NEW_LINE DEDENT else : NEW_LINE INDENT count -= 1 NEW_LINE DEDENT if count == 0 : NEW_LINE INDENT maj_index = i NEW_LINE count = 1 NEW_LINE DEDENT DEDENT return A [ maj_index ] NEW_LINE DEDENT
def isMajority ( A , cand ) : NEW_LINE INDENT count = 0 NEW_LINE for i in range ( len ( A ) ) : NEW_LINE INDENT if A [ i ] == cand : NEW_LINE INDENT count += 1 NEW_LINE DEDENT DEDENT if count > len ( A ) / 2 : NEW_LINE INDENT return True NEW_LINE DEDENT else : NEW_LINE INDENT return False NEW_LINE DEDENT DEDENT
def printMajority ( A ) : NEW_LINE
cand = findCandidate ( A ) NEW_LINE
if isMajority ( A , cand ) == True : NEW_LINE INDENT print ( cand ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ Majority ▁ Element " ) NEW_LINE DEDENT
A = [ 1 , 3 , 3 , 1 , 2 ] NEW_LINE
printMajority ( A ) NEW_LINE
def isSubsetSum ( set , n , sum ) : NEW_LINE
subset = ( [ [ False for i in range ( sum + 1 ) ] for i in range ( n + 1 ) ] ) NEW_LINE
for i in range ( n + 1 ) : NEW_LINE INDENT subset [ i ] [ 0 ] = True NEW_LINE DEDENT
for i in range ( 1 , sum + 1 ) : NEW_LINE INDENT subset [ 0 ] [ i ] = False NEW_LINE DEDENT
for i in range ( 1 , n + 1 ) : NEW_LINE INDENT for j in range ( 1 , sum + 1 ) : NEW_LINE INDENT if j < set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = subset [ i - 1 ] [ j ] NEW_LINE DEDENT if j >= set [ i - 1 ] : NEW_LINE INDENT subset [ i ] [ j ] = ( subset [ i - 1 ] [ j ] or subset [ i - 1 ] [ j - set [ i - 1 ] ] ) NEW_LINE DEDENT DEDENT DEDENT
for i in range ( n + 1 ) : NEW_LINE for j in range ( sum + 1 ) : NEW_LINE print ( subset [ i ] [ j ] , end = " ▁ " ) NEW_LINE print ( ) NEW_LINE return subset [ n ] [ sum ] NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT set = [ 3 , 34 , 4 , 12 , 5 , 2 ] NEW_LINE sum = 9 NEW_LINE n = len ( set ) NEW_LINE if ( isSubsetSum ( set , n , sum ) == True ) : NEW_LINE INDENT print ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " No ▁ subset ▁ with ▁ given ▁ sum " ) NEW_LINE DEDENT DEDENT
def isPowerOfTwo ( x ) : NEW_LINE
return ( x and ( not ( x & ( x - 1 ) ) ) ) NEW_LINE
if ( isPowerOfTwo ( 31 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT if ( isPowerOfTwo ( 64 ) ) : NEW_LINE INDENT print ( ' Yes ' ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( ' No ' ) NEW_LINE DEDENT
def nextPowerOf2 ( n ) : NEW_LINE INDENT count = 0 NEW_LINE DEDENT
if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT n >>= 1 NEW_LINE count += 1 NEW_LINE DEDENT return 1 << count NEW_LINE
n = 0 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def countWays ( n ) : NEW_LINE INDENT res = [ 0 ] * ( n + 2 ) NEW_LINE res [ 0 ] = 1 NEW_LINE res [ 1 ] = 1 NEW_LINE res [ 2 ] = 2 NEW_LINE for i in range ( 3 , n + 1 ) : NEW_LINE INDENT res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] NEW_LINE DEDENT return res [ n ] NEW_LINE DEDENT
n = 4 NEW_LINE print ( countWays ( n ) ) NEW_LINE
def maxTasks ( high , low , n ) : NEW_LINE
if ( n <= 0 ) : NEW_LINE INDENT return 0 NEW_LINE DEDENT
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE DEDENT
OUT = 0 NEW_LINE IN = 1 NEW_LINE
def countWords ( string ) : NEW_LINE INDENT state = OUT NEW_LINE DEDENT
wc = 0 NEW_LINE
for i in range ( len ( string ) ) : NEW_LINE
if ( string [ i ] == ' ▁ ' or string [ i ] ==   ' ' ▁ or ▁ string [ i ] ▁ = = ▁ ' 	 ' ) : NEW_LINE INDENT state = OUT NEW_LINE DEDENT
elif state == OUT : NEW_LINE INDENT state = IN NEW_LINE wc += 1 NEW_LINE DEDENT
return wc NEW_LINE
string =   " One two three NEW_LINE INDENT four five   " NEW_LINE DEDENT print ( " No . ▁ of ▁ words ▁ : ▁ " + str ( countWords ( string ) ) ) NEW_LINE
/ * Returns the maximum among the 2 numbers * / NEW_LINE def max1 ( x , y ) : NEW_LINE INDENT return x if ( x > y ) else y ; NEW_LINE DEDENT
def maxTasks ( high , low , n ) : NEW_LINE
task_dp = [ 0 ] * ( n + 1 ) ; NEW_LINE
task_dp [ 0 ] = 0 ; NEW_LINE
task_dp [ 1 ] = high [ 0 ] ; NEW_LINE
for i in range ( 2 , n + 1 ) : NEW_LINE INDENT task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; NEW_LINE DEDENT return task_dp [ n ] ; NEW_LINE
n = 5 ; NEW_LINE high = [ 3 , 6 , 8 , 7 , 6 ] ; NEW_LINE low = [ 1 , 5 , 4 , 5 , 3 ] ; NEW_LINE print ( maxTasks ( high , low , n ) ) ; NEW_LINE
def findPartition ( arr , n ) : NEW_LINE INDENT sum = 0 NEW_LINE i , j = 0 , 0 NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT if sum % 2 != 0 : NEW_LINE INDENT return false NEW_LINE DEDENT part = [ [ True for i in range ( n + 1 ) ] for j in range ( sum // 2 + 1 ) ] NEW_LINE
for i in range ( 0 , n + 1 ) : NEW_LINE INDENT part [ 0 ] [ i ] = True NEW_LINE DEDENT
for i in range ( 1 , sum // 2 + 1 ) : NEW_LINE INDENT part [ i ] [ 0 ] = False NEW_LINE DEDENT
for i in range ( 1 , sum // 2 + 1 ) : NEW_LINE INDENT for j in range ( 1 , n + 1 ) : NEW_LINE INDENT part [ i ] [ j ] = part [ i ] [ j - 1 ] NEW_LINE if i >= arr [ j - 1 ] : NEW_LINE INDENT part [ i ] [ j ] = ( part [ i ] [ j ] or part [ i - arr [ j - 1 ] ] [ j - 1 ] ) NEW_LINE DEDENT DEDENT DEDENT
return part [ sum // 2 ] [ n ] NEW_LINE
arr = [ 3 , 1 , 1 , 2 , 2 , 1 ] NEW_LINE n = len ( arr ) NEW_LINE
if findPartition ( arr , n ) == True : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two " , " subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " , " two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def start ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT
def state1 ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT
elif ( c == ' h ' or c == ' H ' ) : NEW_LINE INDENT dfa = 2 NEW_LINE DEDENT
else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT
def state2 ( c ) : NEW_LINE
if ( c == ' e ' or c == ' E ' ) : NEW_LINE INDENT dfa = 3 NEW_LINE DEDENT else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT
def state3 ( c ) : NEW_LINE
if ( c == ' t ' or c == ' T ' ) : NEW_LINE INDENT dfa = 1 NEW_LINE DEDENT else : NEW_LINE INDENT dfa = 0 NEW_LINE DEDENT def isAccepted ( string ) : NEW_LINE
length = len ( string ) NEW_LINE for i in range ( length ) : NEW_LINE INDENT if ( dfa == 0 ) : NEW_LINE INDENT start ( string [ i ] ) NEW_LINE DEDENT elif ( dfa == 1 ) : NEW_LINE INDENT state1 ( string [ i ] ) NEW_LINE DEDENT elif ( dfa == 2 ) : NEW_LINE INDENT state2 ( string [ i ] ) NEW_LINE DEDENT else : NEW_LINE INDENT state3 ( string [ i ] ) NEW_LINE DEDENT DEDENT return ( dfa != 3 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT string = " forTHEgeeks " NEW_LINE DEDENT dfa = 0 NEW_LINE INDENT if isAccepted ( string ) : NEW_LINE INDENT print ( " ACCEPTED " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " NOT ▁ ACCEPTED " ) NEW_LINE DEDENT DEDENT
def Series ( n ) : NEW_LINE INDENT sums = 0.0 NEW_LINE for i in range ( 1 , n + 1 ) : NEW_LINE INDENT ser = 1 / ( i ** i ) NEW_LINE sums += ser NEW_LINE DEDENT return sums NEW_LINE DEDENT
n = 3 NEW_LINE res = round ( Series ( n ) , 5 ) NEW_LINE print ( res ) NEW_LINE
import math as mt NEW_LINE
def ternarySearch ( l , r , key , ar ) : NEW_LINE INDENT if ( r >= l ) : NEW_LINE DEDENT
mid1 = l + ( r - l ) // 3 NEW_LINE mid2 = r - ( r - l ) // 3 NEW_LINE
if ( ar [ mid1 ] == key ) : NEW_LINE INDENT return mid1 NEW_LINE DEDENT if ( ar [ mid2 ] == key ) : NEW_LINE INDENT return mid2 NEW_LINE DEDENT
if ( key < ar [ mid1 ] ) : NEW_LINE
return ternarySearch ( l , mid1 - 1 , key , ar ) NEW_LINE elif ( key > ar [ mid2 ] ) : NEW_LINE
return ternarySearch ( mid2 + 1 , r , key , ar ) NEW_LINE else : NEW_LINE
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) NEW_LINE
return - 1 NEW_LINE
l , r , p = 0 , 9 , 5 NEW_LINE
ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] NEW_LINE
l = 0 NEW_LINE
r = 9 NEW_LINE
key = 5 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
key = 50 NEW_LINE
p = ternarySearch ( l , r , key , ar ) NEW_LINE
print ( " Index ▁ of " , key , " is " , p ) NEW_LINE
N = 4 NEW_LINE
def add ( A , B , C ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE C = A [ : ] [ : ] NEW_LINE add ( A , B , C ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( C [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
N = 4 NEW_LINE
def subtract ( A , B , C ) : NEW_LINE INDENT for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] NEW_LINE DEDENT DEDENT DEDENT
A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] NEW_LINE C = A [ : ] [ : ] NEW_LINE subtract ( A , B , C ) NEW_LINE print ( " Result ▁ matrix ▁ is " ) NEW_LINE for i in range ( N ) : NEW_LINE INDENT for j in range ( N ) : NEW_LINE INDENT print ( C [ i ] [ j ] , " ▁ " , end = ' ' ) NEW_LINE DEDENT print ( ) NEW_LINE DEDENT
def linearSearch ( arr , n ) : NEW_LINE INDENT for i in range ( n ) : NEW_LINE INDENT if arr [ i ] is i : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT DEDENT
return - 1 NEW_LINE
arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Fixed ▁ Point ▁ is ▁ " + str ( linearSearch ( arr , n ) ) ) NEW_LINE
def binarySearch ( arr , low , high ) : NEW_LINE INDENT if high >= low : NEW_LINE DEDENT
mid = ( low + high ) // 2 NEW_LINE if mid is arr [ mid ] : NEW_LINE return mid NEW_LINE if mid > arr [ mid ] : NEW_LINE return binarySearch ( arr , ( mid + 1 ) , high ) NEW_LINE else : NEW_LINE return binarySearch ( arr , low , ( mid - 1 ) ) NEW_LINE
return - 1 NEW_LINE
arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Fixed ▁ Point ▁ is ▁ " + str ( binarySearch ( arr , 0 , n - 1 ) ) ) NEW_LINE
def search ( arr , n , x ) : NEW_LINE INDENT for i in range ( 0 , n ) : NEW_LINE INDENT if ( arr [ i ] == x ) : NEW_LINE INDENT return i NEW_LINE DEDENT DEDENT return - 1 NEW_LINE DEDENT
arr = [ 2 , 3 , 4 , 10 , 40 ] NEW_LINE x = 10 NEW_LINE n = len ( arr ) NEW_LINE
result = search ( arr , n , x ) NEW_LINE if ( result == - 1 ) : NEW_LINE INDENT print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Element ▁ is ▁ present ▁ at ▁ index " , result ) NEW_LINE DEDENT
def countSort ( arr ) : NEW_LINE
output = [ 0 for i in range ( len ( arr ) ) ] NEW_LINE
count = [ 0 for i in range ( 256 ) ] NEW_LINE
for i in arr : NEW_LINE INDENT count [ ord ( i ) ] += 1 NEW_LINE DEDENT
for i in range ( 256 ) : NEW_LINE INDENT count [ i ] += count [ i - 1 ] NEW_LINE DEDENT
for i in range ( len ( arr ) ) : NEW_LINE INDENT output [ count [ ord ( arr [ i ] ) ] - 1 ] = arr [ i ] NEW_LINE count [ ord ( arr [ i ] ) ] -= 1 NEW_LINE DEDENT
ans = [ " " for _ in arr ] NEW_LINE for i in range ( len ( arr ) ) : NEW_LINE INDENT ans [ i ] = output [ i ] NEW_LINE DEDENT return ans NEW_LINE
arr = " geeksforgeeks " NEW_LINE ans = countSort ( arr ) NEW_LINE print ( " Sorted ▁ character ▁ array ▁ is ▁ % ▁ s " % ( " " . join ( ans ) ) ) NEW_LINE
def binomialCoeff ( n , k ) : NEW_LINE
if k > n : NEW_LINE INDENT return 0 NEW_LINE DEDENT if k == 0 or k == n : NEW_LINE INDENT return 1 NEW_LINE DEDENT
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) NEW_LINE
n = 5 NEW_LINE k = 2 NEW_LINE print " Value ▁ of ▁ C ( % d , % d ) ▁ is ▁ ( % d ) " % ( n , k , binomialCoeff ( n , k ) ) NEW_LINE
def findoptimal ( N ) : NEW_LINE
if N <= 6 : NEW_LINE INDENT return N NEW_LINE DEDENT
maxi = 0 NEW_LINE
for b in range ( N - 3 , 0 , - 1 ) : NEW_LINE
curr = ( N - b - 1 ) * findoptimal ( b ) NEW_LINE if curr > maxi : NEW_LINE INDENT maxi = curr NEW_LINE DEDENT return maxi NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE
for n in range ( 1 , 21 ) : NEW_LINE INDENT print ( ' Maximum ▁ Number ▁ of ▁ As ▁ with ▁ ' , n , ' keystrokes ▁ is ▁ ' , findoptimal ( n ) ) NEW_LINE DEDENT
def findoptimal ( N ) : NEW_LINE
if ( N <= 6 ) : NEW_LINE INDENT return N NEW_LINE DEDENT
screen = [ 0 ] * N NEW_LINE
for n in range ( 1 , 7 ) : NEW_LINE INDENT screen [ n - 1 ] = n NEW_LINE DEDENT
for n in range ( 7 , N + 1 ) : NEW_LINE
screen [ n - 1 ] = 0 NEW_LINE
for b in range ( n - 3 , 0 , - 1 ) : NEW_LINE
curr = ( n - b - 1 ) * screen [ b - 1 ] NEW_LINE if ( curr > screen [ n - 1 ] ) : NEW_LINE INDENT screen [ n - 1 ] = curr NEW_LINE DEDENT return screen [ N - 1 ] NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE
for N in range ( 1 , 21 ) : NEW_LINE INDENT print ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " , N , " ▁ keystrokes ▁ is ▁ " , findoptimal ( N ) ) NEW_LINE DEDENT
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE elif ( int ( y % 2 ) == 0 ) : NEW_LINE INDENT return ( power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT else : NEW_LINE INDENT return ( x * power ( x , int ( y / 2 ) ) * power ( x , int ( y / 2 ) ) ) NEW_LINE DEDENT DEDENT
x = 2 ; y = 3 NEW_LINE print ( power ( x , y ) ) NEW_LINE
def power ( x , y ) : NEW_LINE INDENT temp = 0 NEW_LINE if ( y == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT temp = power ( x , int ( y / 2 ) ) NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp ; NEW_LINE DEDENT else : NEW_LINE INDENT return x * temp * temp ; NEW_LINE DEDENT DEDENT
def power ( x , y ) : NEW_LINE INDENT if ( y == 0 ) : return 1 NEW_LINE temp = power ( x , int ( y / 2 ) ) NEW_LINE if ( y % 2 == 0 ) : NEW_LINE INDENT return temp * temp NEW_LINE DEDENT else : NEW_LINE INDENT if ( y > 0 ) : return x * temp * temp NEW_LINE else : return ( temp * temp ) / x NEW_LINE DEDENT DEDENT
x , y = 2 , - 3 NEW_LINE print ( ' % .6f ' % ( power ( x , y ) ) ) NEW_LINE
def squareRoot ( n ) : NEW_LINE
x = n NEW_LINE y = 1 NEW_LINE
e = 0.000001 NEW_LINE while ( x - y > e ) : NEW_LINE INDENT x = ( x + y ) / 2 NEW_LINE y = n / x NEW_LINE DEDENT return x NEW_LINE
n = 50 NEW_LINE print ( " Square ▁ root ▁ of " , n , " is " , round ( squareRoot ( n ) , 6 ) ) NEW_LINE
def getAvg ( x , n , sum ) : NEW_LINE INDENT sum = sum + x ; NEW_LINE return float ( sum ) / n ; NEW_LINE DEDENT
def streamAvg ( arr , n ) : NEW_LINE INDENT avg = 0 ; NEW_LINE sum = 0 ; NEW_LINE for i in range ( n ) : NEW_LINE INDENT avg = getAvg ( arr [ i ] , i + 1 , sum ) ; NEW_LINE sum = avg * ( i + 1 ) ; NEW_LINE print ( " Average ▁ of ▁ " , end = " " ) ; NEW_LINE print ( i + 1 , end = " " ) ; NEW_LINE print ( " ▁ numbers ▁ is ▁ " , end = " " ) ; NEW_LINE print ( avg ) ; NEW_LINE DEDENT return ; NEW_LINE DEDENT
arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; NEW_LINE n = len ( arr ) ; NEW_LINE streamAvg ( arr , n ) ; NEW_LINE
def binomialCoefficient ( n , k ) : NEW_LINE INDENT res = 1 NEW_LINE DEDENT
if ( k > n - k ) : NEW_LINE INDENT k = n - k NEW_LINE DEDENT
for i in range ( k ) : NEW_LINE INDENT res = res * ( n - i ) NEW_LINE res = res / ( i + 1 ) NEW_LINE DEDENT return res NEW_LINE
n = 8 NEW_LINE k = 2 NEW_LINE res = binomialCoefficient ( n , k ) NEW_LINE print ( " Value ▁ of ▁ C ( % ▁ d , ▁ % ▁ d ) ▁ is ▁ % ▁ d " % ( n , k , res ) ) NEW_LINE
import math NEW_LINE
def primeFactors ( n ) : NEW_LINE
while n % 2 == 0 : NEW_LINE INDENT print 2 , NEW_LINE n = n / 2 NEW_LINE DEDENT
for i in range ( 3 , int ( math . sqrt ( n ) ) + 1 , 2 ) : NEW_LINE
while n % i == 0 : NEW_LINE INDENT print i , NEW_LINE n = n / i NEW_LINE DEDENT
if n > 2 : NEW_LINE INDENT print n NEW_LINE DEDENT
n = 315 NEW_LINE primeFactors ( n ) NEW_LINE
def printCombination ( arr , n , r ) : NEW_LINE
data = [ 0 ] * r ; NEW_LINE
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; NEW_LINE
def combinationUtil ( arr , data , start , end , index , r ) : NEW_LINE
if ( index == r ) : NEW_LINE INDENT for j in range ( r ) : NEW_LINE INDENT print ( data [ j ] , end = " ▁ " ) ; NEW_LINE DEDENT print ( ) ; NEW_LINE return ; NEW_LINE DEDENT
i = start ; NEW_LINE while ( i <= end and end - i + 1 >= r - index ) : NEW_LINE INDENT data [ index ] = arr [ i ] ; NEW_LINE combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; NEW_LINE i += 1 ; NEW_LINE DEDENT
arr = [ 1 , 2 , 3 , 4 , 5 ] ; NEW_LINE r = 3 ; NEW_LINE n = len ( arr ) ; NEW_LINE printCombination ( arr , n , r ) ; NEW_LINE
def printCombination ( arr , n , r ) : NEW_LINE
data = [ 0 ] * r NEW_LINE
combinationUtil ( arr , n , r , 0 , data , 0 ) NEW_LINE
def combinationUtil ( arr , n , r , index , data , i ) : NEW_LINE
if ( index == r ) : NEW_LINE INDENT for j in range ( r ) : NEW_LINE INDENT print ( data [ j ] , end = " ▁ " ) NEW_LINE DEDENT print ( ) NEW_LINE return NEW_LINE DEDENT
if ( i >= n ) : NEW_LINE INDENT return NEW_LINE DEDENT
data [ index ] = arr [ i ] NEW_LINE combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) NEW_LINE
combinationUtil ( arr , n , r , index , data , i + 1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ 1 , 2 , 3 , 4 , 5 ] NEW_LINE r = 3 NEW_LINE n = len ( arr ) NEW_LINE printCombination ( arr , n , r ) NEW_LINE DEDENT
def findgroups ( arr , n ) : NEW_LINE
c = [ 0 , 0 , 0 ] NEW_LINE
res = 0 NEW_LINE
for i in range ( 0 , n ) : NEW_LINE INDENT c [ arr [ i ] % 3 ] += 1 NEW_LINE DEDENT
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) NEW_LINE
res += c [ 1 ] * c [ 2 ] NEW_LINE
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * ( c [ 0 ] - 2 ) ) / 6 NEW_LINE
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * ( c [ 1 ] - 2 ) ) / 6 NEW_LINE
res += ( ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) NEW_LINE
res += c [ 0 ] * c [ 1 ] * c [ 2 ] NEW_LINE
return res NEW_LINE
arr = [ 3 , 6 , 7 , 2 , 9 ] NEW_LINE n = len ( arr ) NEW_LINE print ( " Required ▁ number ▁ of ▁ groups ▁ are " , int ( findgroups ( arr , n ) ) ) NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT count = 0 ; NEW_LINE DEDENT
if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( n != 0 ) : NEW_LINE INDENT n >>= 1 NEW_LINE count += 1 NEW_LINE DEDENT return 1 << count ; NEW_LINE
n = 0 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT p = 1 NEW_LINE if ( n and not ( n & ( n - 1 ) ) ) : NEW_LINE INDENT return n NEW_LINE DEDENT while ( p < n ) : NEW_LINE INDENT p <<= 1 NEW_LINE DEDENT return p ; NEW_LINE DEDENT
n = 5 NEW_LINE print ( nextPowerOf2 ( n ) ) ; NEW_LINE
def nextPowerOf2 ( n ) : NEW_LINE INDENT n -= 1 NEW_LINE n |= n >> 1 NEW_LINE n |= n >> 2 NEW_LINE n |= n >> 4 NEW_LINE n |= n >> 8 NEW_LINE n |= n >> 16 NEW_LINE n += 1 NEW_LINE return n NEW_LINE DEDENT
n = 5 NEW_LINE print ( nextPowerOf2 ( n ) ) NEW_LINE
def segregate0and1 ( arr , size ) : NEW_LINE
left , right = 0 , size - 1 NEW_LINE while left < right : NEW_LINE
while arr [ left ] == 0 and left < right : NEW_LINE INDENT left += 1 NEW_LINE DEDENT
while arr [ right ] == 1 and left < right : NEW_LINE INDENT right -= 1 NEW_LINE DEDENT
if left < right : NEW_LINE INDENT arr [ left ] = 0 NEW_LINE arr [ right ] = 1 NEW_LINE left += 1 NEW_LINE right -= 1 NEW_LINE DEDENT return arr NEW_LINE
arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] NEW_LINE arr_size = len ( arr ) NEW_LINE print ( " Array ▁ after ▁ segregation " ) NEW_LINE print ( segregate0and1 ( arr , arr_size ) ) NEW_LINE
def maxIndexDiff ( arr , n ) : NEW_LINE INDENT maxDiff = - 1 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT j = n - 1 NEW_LINE while ( j > i ) : NEW_LINE INDENT if arr [ j ] > arr [ i ] and maxDiff < ( j - i ) : NEW_LINE INDENT maxDiff = j - i NEW_LINE DEDENT j -= 1 NEW_LINE DEDENT DEDENT return maxDiff NEW_LINE DEDENT
arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] NEW_LINE n = len ( arr ) NEW_LINE maxDiff = maxIndexDiff ( arr , n ) NEW_LINE print ( maxDiff ) NEW_LINE
def findStep ( n ) : NEW_LINE INDENT if ( n == 1 or n == 0 ) : NEW_LINE INDENT return 1 NEW_LINE DEDENT elif ( n == 2 ) : NEW_LINE INDENT return 2 NEW_LINE DEDENT else : NEW_LINE INDENT return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) NEW_LINE DEDENT DEDENT
n = 4 NEW_LINE print ( findStep ( n ) ) NEW_LINE
def isSubsetSum ( arr , n , sum ) : NEW_LINE
if sum == 0 : NEW_LINE INDENT return True NEW_LINE DEDENT if n == 0 and sum != 0 : NEW_LINE INDENT return False NEW_LINE DEDENT
if arr [ n - 1 ] > sum : NEW_LINE INDENT return isSubsetSum ( arr , n - 1 , sum ) NEW_LINE DEDENT
return isSubsetSum ( arr , n - 1 , sum ) or isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) NEW_LINE
def findPartion ( arr , n ) : NEW_LINE
sum = 0 NEW_LINE for i in range ( 0 , n ) : NEW_LINE INDENT sum += arr [ i ] NEW_LINE DEDENT
if sum % 2 != 0 : NEW_LINE INDENT return false NEW_LINE DEDENT
return isSubsetSum ( arr , n , sum // 2 ) NEW_LINE
arr = [ 3 , 1 , 5 , 9 , 12 ] NEW_LINE n = len ( arr ) NEW_LINE
if findPartion ( arr , n ) == True : NEW_LINE INDENT print ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) NEW_LINE DEDENT
def findRepeatFirstN2 ( s ) : NEW_LINE
p = - 1 NEW_LINE for i in range ( len ( s ) ) : NEW_LINE INDENT for j in range ( i + 1 , len ( s ) ) : NEW_LINE INDENT if ( s [ i ] == s [ j ] ) : NEW_LINE INDENT p = i NEW_LINE break NEW_LINE DEDENT DEDENT if ( p != - 1 ) : NEW_LINE INDENT break NEW_LINE DEDENT DEDENT return p NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT str = " geeksforgeeks " NEW_LINE pos = findRepeatFirstN2 ( str ) NEW_LINE if ( pos == - 1 ) : NEW_LINE INDENT print ( " Not ▁ found " ) NEW_LINE DEDENT else : NEW_LINE INDENT print ( str [ pos ] ) NEW_LINE DEDENT DEDENT
def isPalindrome ( num ) : NEW_LINE INDENT reverse_num = 0 NEW_LINE DEDENT
temp = num NEW_LINE while ( temp != 0 ) : NEW_LINE INDENT remainder = temp % 10 NEW_LINE reverse_num = reverse_num * 10 + remainder NEW_LINE temp = int ( temp / 10 ) NEW_LINE DEDENT
if ( reverse_num == num ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE
def isOddLength ( num ) : NEW_LINE INDENT count = 0 NEW_LINE while ( num > 0 ) : NEW_LINE INDENT num = int ( num / 10 ) NEW_LINE count += 1 NEW_LINE DEDENT if ( count % 2 != 0 ) : NEW_LINE INDENT return True NEW_LINE DEDENT return False NEW_LINE DEDENT
def sumOfAllPalindrome ( L , R ) : NEW_LINE INDENT sum = 0 NEW_LINE if ( L <= R ) : NEW_LINE INDENT for i in range ( L , R + 1 , 1 ) : NEW_LINE DEDENT DEDENT
if ( isPalindrome ( i ) and isOddLength ( i ) ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT L = 110 NEW_LINE R = 1130 NEW_LINE print ( sumOfAllPalindrome ( L , R ) ) NEW_LINE DEDENT
def subtractOne ( x ) : NEW_LINE INDENT m = 1 NEW_LINE DEDENT
while ( ( x & m ) == False ) : NEW_LINE INDENT x = x ^ m NEW_LINE m = m << 1 NEW_LINE DEDENT
x = x ^ m NEW_LINE return x NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT print ( subtractOne ( 13 ) ) NEW_LINE DEDENT
def findSum ( n , a , b ) : NEW_LINE INDENT sum = 0 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE DEDENT
if ( i % a == 0 or i % b == 0 ) : NEW_LINE INDENT sum += i NEW_LINE DEDENT return sum NEW_LINE
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT n = 10 NEW_LINE a = 3 NEW_LINE b = 5 NEW_LINE print ( findSum ( n , a , b ) ) NEW_LINE DEDENT
def pell ( n ) : NEW_LINE INDENT if ( n <= 2 ) : NEW_LINE INDENT return n NEW_LINE DEDENT return ( 2 * pell ( n - 1 ) + pell ( n - 2 ) ) NEW_LINE DEDENT
n = 4 ; NEW_LINE print ( pell ( n ) ) NEW_LINE
def largestPower ( n , p ) : NEW_LINE
x = 0 NEW_LINE
while n : NEW_LINE INDENT n /= p NEW_LINE x += n NEW_LINE DEDENT return x NEW_LINE
n = 10 ; p = 3 NEW_LINE print ( " The largest power of % d that divides % d ! is % d " % ( p , n , largestPower ( n , p ) ) ) NEW_LINE
def factorial ( n ) : NEW_LINE
return 1 if ( n == 1 or n == 0 ) else n * factorial ( n - 1 ) NEW_LINE
num = 5 NEW_LINE print ( " Factorial ▁ of " , num , " is " , factorial ( num ) ) NEW_LINE
def bitExtracted ( number , k , p ) : NEW_LINE INDENT return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; NEW_LINE DEDENT
number = 171 NEW_LINE k = 5 NEW_LINE p = 2 NEW_LINE print " The ▁ extracted ▁ number ▁ is ▁ " , bitExtracted ( number , k , p ) NEW_LINE
import sys NEW_LINE
def solve ( a , n ) : NEW_LINE INDENT max1 = - sys . maxsize - 1 NEW_LINE for i in range ( 0 , n , 1 ) : NEW_LINE INDENT for j in range ( 0 , n , 1 ) : NEW_LINE INDENT if ( abs ( a [ i ] - a [ j ] ) > max1 ) : NEW_LINE INDENT max1 = abs ( a [ i ] - a [ j ] ) NEW_LINE DEDENT DEDENT DEDENT return max1 NEW_LINE DEDENT
if __name__ == ' _ _ main _ _ ' : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : " , solve ( arr , size ) ) NEW_LINE DEDENT
def solve ( a , n ) : NEW_LINE INDENT min1 = a [ 0 ] NEW_LINE max1 = a [ 0 ] NEW_LINE DEDENT
for i in range ( n ) : NEW_LINE INDENT if ( a [ i ] > max1 ) : NEW_LINE INDENT max1 = a [ i ] NEW_LINE DEDENT if ( a [ i ] < min1 ) : NEW_LINE INDENT min1 = a [ i ] NEW_LINE DEDENT DEDENT return abs ( min1 - max1 ) NEW_LINE
if __name__ == " _ _ main _ _ " : NEW_LINE INDENT arr = [ - 1 , 2 , 3 , 4 , - 10 ] NEW_LINE size = len ( arr ) NEW_LINE print ( " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( arr , size ) ) NEW_LINE DEDENT
